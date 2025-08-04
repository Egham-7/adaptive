# llm_core_models.py


from typing import Any

from openai.types.chat import (
    CompletionCreateParams,
)
from pydantic import BaseModel, Field

from .llm_enums import ProviderType, TaskType  # Import ProviderType for TaskModelEntry


class ModelCapability(BaseModel):
    description: str | None = None
    provider: Any = None  # Accept any provider string or ProviderType
    model_name: str
    cost_per_1m_input_tokens: float | None = Field(
        None, alias="cost_per_1m_input_tokens"
    )
    cost_per_1m_output_tokens: float | None = Field(
        None, alias="cost_per_1m_output_tokens"
    )
    max_context_tokens: int | None = Field(None, alias="max_context_tokens")
    max_output_tokens: int | None = Field(None, alias="max_output_tokens")
    supports_function_calling: bool | None = Field(
        None, alias="supports_function_calling"
    )
    languages_supported: list[str] = Field(
        default_factory=list, alias="languages_supported"
    )
    model_size_params: str | None = Field(None, alias="model_size_params")
    latency_tier: str | None = Field(None, alias="latency_tier")

    # NEW: Task-specific capabilities for custom models
    task_type: str | None = Field(
        None, alias="task_type"
    )  # "OPEN_QA", "CODE_GENERATION", etc.
    complexity: str | None = Field(None, alias="complexity")  # "easy", "medium", "hard"


class ProtocolManagerConfig(BaseModel):
    """Configuration for the protocol manager"""

    models: list[ModelCapability] | None = None
    cost_bias: float | None = None
    complexity_threshold: float | None = None
    token_threshold: int | None = None


class ModelEntry(BaseModel):
    providers: list[ProviderType | str]
    model_name: str = Field(alias="model_name")
    # Remove _original_provider since we no longer need fallback metadata


class TaskModelMapping(BaseModel):
    model_entries: list[ModelEntry] = Field(alias="model_entries")


class ModelSelectionConfig(BaseModel):
    model_capabilities: dict[str, ModelCapability] = Field(alias="model_capabilities")
    task_model_mappings: dict[TaskType, TaskModelMapping] = Field(
        alias="task_model_mappings"
    )


class ModelSelectionRequest(BaseModel):
    """
    Model selection request that contains OpenAI's CompletionCreateParams
    plus our custom model selection parameters.
    """

    # The OpenAI chat completion request
    chat_completion_request: CompletionCreateParams

    # Our custom parameters for model selection
    user_id: str | None = None
    protocol_manager_config: ProtocolManagerConfig | None = None
