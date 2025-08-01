# llm_core_models.py


from openai.types.chat import (
    CompletionCreateParams,
)
from pydantic import BaseModel, Field, model_validator

from .llm_enums import ProviderType, TaskType  # Import ProviderType for TaskModelEntry


class ModelCapability(BaseModel):
    description: str | None = None
    provider: ProviderType
    model_name: str
    cost_per_1m_input_tokens: float = Field(alias="cost_per_1m_input_tokens")
    cost_per_1m_output_tokens: float = Field(alias="cost_per_1m_output_tokens")
    max_context_tokens: int = Field(alias="max_context_tokens")
    max_output_tokens: int | None = Field(None, alias="max_output_tokens")
    supports_function_calling: bool = Field(alias="supports_function_calling")
    languages_supported: list[str] = Field(
        default_factory=list, alias="languages_supported"
    )
    model_size_params: str | None = Field(None, alias="model_size_params")
    latency_tier: str | None = Field(None, alias="latency_tier")


class ProtocolManagerConfig(BaseModel):
    """Configuration for the protocol manager"""

    models: list[ModelCapability] | None = None
    cost_bias: float | None = None
    complexity_threshold: float | None = None
    token_threshold: int | None = None


class ModelEntry(BaseModel):
    providers: list[ProviderType]
    model_name: str = Field(alias="model_name")


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

    @model_validator(mode="after")
    def validate_parameters(self) -> "ModelSelectionRequest":
        # Validate the OpenAI request has required fields
        if not self.chat_completion_request.get("messages"):
            raise ValueError("messages cannot be empty")

        # Ensure protocol_manager_config exists
        if not self.protocol_manager_config:
            self.protocol_manager_config = ProtocolManagerConfig()

        return self
