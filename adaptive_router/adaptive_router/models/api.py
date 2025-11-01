"""Public API models for model selection requests and responses.

This module contains the public-facing API models that external users interact with
when making model selection requests to the adaptive router service.
"""

from typing import Any

from pydantic import BaseModel, field_validator

from adaptive_router.models.registry import RegistryModel


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

    models: list[RegistryModel] | None = None
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


class Alternative(BaseModel):
    """Alternative model option for routing.

    Attributes:
        provider: Model provider
        model: Model name
    """

    provider: str
    model: str

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Provider cannot be empty or whitespace only")
        return v.strip()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model cannot be empty or whitespace only")
        return v.strip()


class ModelSelectionResponse(BaseModel):
    """Simplified response with just the selected model and alternatives.

    Attributes:
        provider: Selected model provider
        model: Selected model name
        alternatives: List of alternative model options
    """

    provider: str
    model: str
    alternatives: list[Alternative]

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Provider cannot be empty or whitespace only")
        return v.strip()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model cannot be empty or whitespace only")
        return v.strip()
