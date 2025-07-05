# llm_core_models.py

from typing import Any, Literal

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


class TaskModelEntry(BaseModel):
    provider: ProviderType
    model_name: str = Field(alias="model_name")


class TaskModelMapping(BaseModel):
    model_entries: list[TaskModelEntry] = Field(alias="model_entries")


class ModelSelectionConfig(BaseModel):
    model_capabilities: dict[str, ModelCapability] = Field(alias="model_capabilities")
    task_model_mappings: dict[TaskType, TaskModelMapping] = Field(
        alias="task_model_mappings"
    )


class ImageUrl(BaseModel):
    url: str
    detail: Literal["low", "high", "auto"] | None = "auto"


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None

    @model_validator(mode="after")
    def validate_content(self) -> "ContentItem":
        if self.type == "text" and not self.text:
            raise ValueError("text must be provided when type is 'text'")
        if self.type == "image_url" and not self.image_url:
            raise ValueError("image_url must be provided when type is 'image_url'")
        return self


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[ContentItem] | None = None
    name: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> "Message":
        # Content is required for system, user messages
        if self.role in ["system", "user"] and not self.content:
            raise ValueError(f"content is required for {self.role} messages")

        # Tool messages need tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool_call_id is required for tool messages")

        # Function messages need name
        if self.role == "function" and not self.name:
            raise ValueError("name is required for function messages")

        return self

    def get_text_content(self) -> str:
        """Extract text content from message for backward compatibility."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = [
                item.text for item in self.content if item.type == "text" and item.text
            ]
            return " ".join(text_parts)
        return ""


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: Literal["function"]
    function: Function


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"]


class ModelSelectionRequest(BaseModel):
    # Required
    messages: list[Message]

    # Model selection (existing)
    provider_constraint: list[str] | None = Field(
        default=None, alias="provider_constraint"
    )
    cost_bias: float | None = Field(default=None, alias="cost_bias")

    # OpenAI Chat Completion parameters
    model: str | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    max_tokens: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=1, ge=1, le=128)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    user: str | None = None

    # Legacy function calling (deprecated in favor of tools)
    functions: list[Function] | None = None
    function_call: str | dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "ModelSelectionRequest":
        # Validate logprobs configuration
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("logprobs must be True when top_logprobs is specified")

        # Validate function vs tools usage
        if self.functions and self.tools:
            raise ValueError(
                "Cannot specify both functions and tools - use tools instead"
            )

        # Validate tool_choice with tools
        if self.tool_choice and not self.tools:
            raise ValueError("tool_choice requires tools to be specified")

        # Validate function_call with functions
        if self.function_call and not self.functions:
            raise ValueError("function_call requires functions to be specified")

        # Validate messages is not empty
        if not self.messages:
            raise ValueError("messages cannot be empty")

        return self

    @property
    def prompt(self) -> str:
        """Convert messages to a single prompt string for backward compatibility."""
        return "\n".join(
            [f"{msg.role}: {msg.get_text_content()}" for msg in self.messages]
        )
