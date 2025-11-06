"""Public API models for model selection requests and responses.

This module contains the public-facing API models that external users interact with
when making model selection requests to the adaptive router service.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Model(BaseModel):
    """Model specification for routing with cost information.

    Contains the essential fields needed for model identification
    and routing decisions, including mandatory cost data.

    Attributes:
        provider: Model provider (e.g., "openai", "anthropic")
        model_name: Model name (e.g., "gpt-4", "claude-sonnet-4-5")
        cost_per_1m_input_tokens: Cost per 1M input tokens
        cost_per_1m_output_tokens: Cost per 1M output tokens
    """

    provider: str
    model_name: str
    cost_per_1m_input_tokens: float = Field(
        ..., gt=0, description="Cost per 1M input tokens"
    )
    cost_per_1m_output_tokens: float = Field(
        ..., gt=0, description="Cost per 1M output tokens"
    )

    @property
    def cost_per_1m_tokens(self) -> float:
        """Average cost per million tokens (for routing calculations)."""
        return (self.cost_per_1m_input_tokens + self.cost_per_1m_output_tokens) / 2.0

    def unique_id(self) -> str:
        """Construct the router-compatible unique identifier.

        Returns:
            Unique identifier in format "provider:model_name"

        Raises:
            ValueError: If provider or model_name is empty
        """
        provider = (self.provider or "").strip().lower()
        if not provider:
            raise ValueError("Model missing provider field")
        
        model_name = self.model_name.strip().lower()
        if not model_name:
            raise ValueError(f"Model '{provider}' missing model_name")
        
        return f"{provider}:{model_name}"


class ModelSelectionRequest(BaseModel):
    """Model selection request for intelligent routing.

    Contains the prompt and context information needed for intelligent model
    routing, including tool usage detection and user preferences.

    Attributes:
        prompt: The user prompt to analyze
        tool_call: Current tool call being made (for function calling detection)
        tools: Available tool definitions
        user_id: User identifier for tracking
        models: Optional list of registry models to restrict routing to
        cost_bias: Cost preference (0.0=cheap, 1.0=quality)
        complexity_threshold: Complexity threshold for model selection
        token_threshold: Token count threshold for model selection
    """

    # The user prompt to analyze
    prompt: str

    # Tool-related fields for function calling detection
    tool_call: dict[str, Any] | None = None  # Current tool call being made
    tools: list[dict[str, Any]] | None = None  # Available tool definitions

    # Our custom parameters for model selection
    user_id: str | None = None

    models: list[Model] | None = None
    cost_bias: float | None = None
    complexity_threshold: float | None = None
    token_threshold: int | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("cost_bias")
    @classmethod
    def validate_cost_bias(cls, v: float | None) -> float | None:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Cost bias must be between 0.0 and 1.0")
        return v


class ModelSelectionAPIRequest(BaseModel):
    """API request model that accepts model specifications as strings.

    This is the external API model that accepts "provider:model_name" strings,
    which are then resolved to Model objects internally.
    """

    prompt: str
    user_id: str | None = None
    models: list[str] | None = None
    cost_bias: float | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("cost_bias")
    @classmethod
    def validate_cost_bias(cls, v: float | None) -> float | None:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Cost bias must be between 0.0 and 1.0")
        return v

    def to_internal_request(
        self, resolved_models: list[Model] | None
    ) -> ModelSelectionRequest:
        """Convert to internal ModelSelectionRequest with resolved models."""
        return ModelSelectionRequest(
            prompt=self.prompt,
            user_id=self.user_id,
            models=resolved_models,
            cost_bias=self.cost_bias,
        )


class Alternative(BaseModel):
    """Alternative model option for routing.

    Attributes:
        model_id: Model identifier (e.g., "anthropic:claude-sonnet-4-5")
    """

    model_id: str

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model ID cannot be empty or whitespace only")
        return v.strip()


class ModelSelectionResponse(BaseModel):
    """Clean response with selected model and alternatives.

    Attributes:
        model_id: Selected model identifier (e.g., "anthropic:claude-sonnet-4-5")
        alternatives: List of alternative model options
    """

    model_id: str
    alternatives: list[Alternative]

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model ID cannot be empty or whitespace only")
        return v.strip()
