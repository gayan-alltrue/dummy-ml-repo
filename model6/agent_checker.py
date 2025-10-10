import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, TypeVar

import logfire
from alltrue.local.ai_agents.ai_agent_factory import AiModel, get_agent
from pydantic import BaseModel
from pydantic_ai import Agent

from app.core.model_discovery.schemas.enumerators import FileType
from app.core.model_discovery.schemas.model_card import ModelCardAgentOutput
from app.core.model_discovery.utils.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    ModelCardValidationError,
)

# Environment-based configuration
AI_MODEL = os.getenv("OPENAI_MODEL", "o3")
MAX_BYTES: int = int(os.getenv("MAX_BYTES", 10 * 1024 * 1024))
INSTRUMENT: bool = False

# Type variable for Pydantic model outputs
T = TypeVar("T", bound=BaseModel)


class IChecker(ABC):
    @abstractmethod
    async def check(self, **kwargs: Any) -> Tuple[FileType, dict | None]:
        ...


class AgentHandler:
    @staticmethod
    def get_agent_object(prompt: str, output_type: Type[T]) -> Agent[T]:
        if not prompt:
            raise ConfigurationError("Agent prompt is not provided")

        if not output_type:
            raise ConfigurationError("Agent output type is not provided")

        try:
            return get_agent(
                AiModel.BEDROCK_PRIMARY_MODEL,
                system_prompt=prompt,
                output_type=output_type,
            )
        except Exception as e:
            raise AgentExecutionError(f"Failed to create agent: {e}") from e


class BaseAgentChecker(IChecker):
    """Base class for agent-based checkers with common functionality."""

    def __init__(self, prompt: str, output_type: Type[T]):
        self.agent: Agent[T] = AgentHandler.get_agent_object(prompt, output_type)

    async def _execute_agent(self, input_data: str) -> Any:
        if not input_data:
            raise AgentExecutionError("Input data is not provided for agent execution")

        try:
            if asyncio.iscoroutinefunction(self.agent.run):
                return await self.agent.run(input_data)
            else:
                return await asyncio.to_thread(self.agent.run, input_data)
        except Exception as e:
            logfire.error(f"Agent execution failed: {e}")
            raise AgentExecutionError(f"Agent execution failed: {e}") from e

    def _get_output_field(
        self, result: Any, field_name: str, default: Any = None
    ) -> Any:
        if not result:
            return default

        if not hasattr(result, "output"):
            logfire.warning(f"Agent result does not have 'output' attribute")
            return default

        return getattr(result.output, field_name, default)


class ModelArtifactChecker(IChecker):
    """Simple checker that validates against confirmed artifacts."""

    @logfire.instrument
    async def check(self, **kwargs: Any) -> Tuple[FileType, dict | None]:
        file_name: str = kwargs.get("file_name", "")
        confirmed_artifacts: Dict[str, str] = kwargs.get("confirmed_artifacts", {})
        logfire.info(f"Checking {file_name} against confirmed artifacts")
        logfire.info(f"{confirmed_artifacts}")
        try:
            ft = FileType(confirmed_artifacts.get(file_name, FileType.OTHER.value))
            logfire.info(f"Detected {file_name} as {ft.value}")
            return ft, None
        except Exception as e:
            logfire.warning(f"Model artifact detection failed: {e}")
            return FileType.OTHER, None


class ModelCardChecker(BaseAgentChecker):
    """Checker for model card detection using AI agent."""

    def __init__(self):
        prompt = (
            "You will be given text that may or may not be a model card. "
            "Your task is two fold: "
            "1. Determine if the text is a model card or not. Output goes in 'is_model_card'. "
            "2. If it is, structure it into 'model_card_data'."
        )
        super().__init__(prompt, ModelCardAgentOutput)

    @logfire.instrument
    async def check(self, **kwargs: Any) -> Tuple[FileType, dict | None]:
        model_card_text = kwargs.get("model_card_text")
        file_name: str = kwargs.get("file_name", "")

        if not model_card_text:
            raise ModelCardValidationError(
                "Model card text is not provided for model card checking"
            )

        if not file_name:
            raise ModelCardValidationError(
                "File name is not provided for model card checking"
            )

        logfire.info(f"Checking {file_name} for model card detection")

        try:

            result = await self._execute_agent(model_card_text)

            if result and self._get_output_field(result, "is_model_card", False):
                logfire.info(f"Detected model card in {file_name}")
                logfire.debug(f"Model card text length: {len(model_card_text)} bytes")
                return (
                    FileType.MODEL_CARD,
                    self._get_output_field(result, "model_card_data", None),
                )
            logfire.warning(f"{file_name} is not a modelcard")
            logfire.debug(f"Non model card text length: {len(model_card_text)} bytes")
            return FileType.OTHER, None

        except Exception as e:
            logfire.error(f"Failed to check model card for {file_name}: {e}")
            raise ModelCardValidationError(
                f"Failed to check model card for {file_name}: {e}"
            ) from e


class ModelWeightListAgentOutput(BaseModel):
    filtered_files: List[str]


class ModelWeightBatchChecker(BaseAgentChecker):
    """
    Batches a list of filenames through a cached Agent instance
    and returns only those that look like model-weight files.
    """

    PROMPT = (
        "You will receive a JSON array of filenames (strings).\n"
        "Your task is to:\n"
        "1. From the list, select only those filenames that are likely to be:\n"
        "   - Machine learning model weight files\n"
        "   - Executable pickle files (e.g., files containing serialized Python objects)\n"
        "2. Use only the filename and extension to make this decision.\n"
        "   Examples of matching filenames/extensions include:\n"
        "     - model.pt, model.h5, model.onnx\n"
        "     - pytorch_model.bin, tf_model.ckpt\n"
        "     - weights.pb, checkpoint.ckpt\n"
        "     - model.pkl, data.pkl, suspicious_model.sav\n"
        "     - model.safetensors, model.keras\n"
        "     - anything ending with .pkl, .pt, .h5, .ckpt, .bin, .onnx, .sav, .pb, .safetensors, .keras, .pickle.dat, .pickle, .engine, .llamafile, .so, .engine, .dat, .gguf\n\n"
        "3. Return ONLY a valid JSON object with exactly this shape:\n"
        '   {"filtered_files": ["file1.pt", "file2.pkl"]}\n\n'
        "Do not include any other text, explanation, or markdown. Respond with valid JSON only."
    )

    def __init__(
        self,
        batch_size: int = 50,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        super().__init__(self.PROMPT, ModelWeightListAgentOutput)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @logfire.instrument
    async def _call_agent_with_retry(self, batch: List[str]) -> List[str]:
        if not batch:
            logfire.warning("Empty batch provided for agent processing")
            return []

        try:
            payload = json.dumps(batch)
        except Exception as e:
            raise AgentExecutionError(f"Failed to serialize batch to JSON: {e}") from e

        for attempt in range(1, self.max_retries + 1):
            logfire.info(f"Attempt {attempt} for batch: {batch}")
            try:
                result = await self._execute_agent(payload)
                filtered = self._get_output_field(result, "filtered_files", [])

                if filtered:
                    logfire.info(f"Batch succeeded on attempt {attempt}: {filtered}")
                    return filtered

            except Exception as e:
                logfire.error(f"Agent batch failed on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise AgentExecutionError(
                        f"Agent batch failed after {self.max_retries} attempts: {e}"
                    ) from e

                retry_wait = self.retry_delay * (2 ** (attempt - 1))
                logfire.warning(f"Retrying in {retry_wait} seconds...")
                await asyncio.sleep(retry_wait)

        return []

    @logfire.instrument
    async def check(self, **kwargs: Any) -> Tuple[FileType, dict | None]:
        all_names: List[str] = kwargs.get("all_artifact_names", [])

        if not all_names:
            logfire.warning("No artifact names provided for batch checking")
            return FileType.ARTIFACT, {}

        results: Dict[str, str] = {}

        try:
            for i in range(0, len(all_names), self.batch_size):
                batch = all_names[i : i + self.batch_size]
                filtered = await self._call_agent_with_retry(batch)
                for name in filtered:
                    results[name] = FileType.ARTIFACT.value

            return FileType.ARTIFACT, results
        except Exception as e:
            logfire.error(f"Failed to check artifacts in batch: {e}")
            raise AgentExecutionError(f"Failed to check artifacts in batch: {e}") from e
