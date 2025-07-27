# llm_core_models.py


from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    CompletionCreateParams,
)
from pydantic import BaseModel, Field, model_validator

from .llm_enums import ProviderType, TaskType  # Import ProviderType for TaskModelEntry


class ModelCapability(BaseModel):
    description: str
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
    provider_constraint: list[str] | None = None
    cost_bias: float | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "ModelSelectionRequest":
        # Validate the OpenAI request has required fields
        if not self.chat_completion_request.get("messages"):
            raise ValueError("messages cannot be empty")

        return self

    @property
    def tools(self) -> list[ChatCompletionToolParam] | None:
        """Access tools from the OpenAI request."""
        from typing import cast

        return cast(
            list[ChatCompletionToolParam] | None,
            self.chat_completion_request.get("tools"),
        )

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        """Access messages from the OpenAI request."""
        from typing import cast

        return cast(
            list[ChatCompletionMessageParam], self.chat_completion_request["messages"]
        )

    @property
    def prompt(self) -> str:
        """Convert messages to a single prompt string for backward compatibility."""
        text_parts = []
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text_content = " ".join(
                    [
                        item.get("text", "")
                        for item in content
                        if item.get("type") == "text" and item.get("text")
                    ]
                )
                if text_content:
                    text_parts.append(f"{role}: {text_content}")
        return "\n".join(text_parts)
