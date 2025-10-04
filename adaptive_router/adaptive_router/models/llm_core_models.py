from typing import Any

from pydantic import BaseModel, field_validator


class ModelCapability(BaseModel):
    """Unified model capability supporting both partial and full specifications."""

    # Required fields
    provider: str | None = None
    model_name: str | None = None

    # Optional fields (for partial specs)
    cost_per_1m_input_tokens: float | None = None
    cost_per_1m_output_tokens: float | None = None
    max_context_tokens: int | None = None
    supports_function_calling: bool | None = None

    # Task info
    task_type: str | None = None
    complexity: str | None = None  # "easy", "medium", "hard"

    # Metadata
    description: str | None = None
    languages_supported: list[str] | None = None
    experimental: bool | None = None

    @property
    def is_partial(self) -> bool:
        """True if this is a partial specification missing required fields."""
        return any(
            field is None
            for field in [
                self.provider,
                self.model_name,
                self.cost_per_1m_input_tokens,
                self.cost_per_1m_output_tokens,
                self.max_context_tokens,
                self.supports_function_calling,
            ]
        )

    @property
    def unique_id(self) -> str:
        """Unique identifier for this model with normalized case."""
        if not self.provider:
            raise ValueError("Provider is required for unique_id")
        if not self.model_name:
            raise ValueError("Model name is required for unique_id")
        return f"{self.provider.lower()}:{self.model_name.lower()}"

    @property
    def complexity_score(self) -> float:
        """Convert string complexity to numeric score (0.0-1.0).

        Returns:
            0.2 for "easy", 0.5 for "medium", 0.8 for "hard", 0.5 for None/unknown
        """
        if not self.complexity:
            return 0.5  # Default for unknown complexity

        complexity_lower = self.complexity.lower().strip()
        complexity_mapping = {
            "easy": 0.2,
            "medium": 0.5,
            "hard": 0.8,
        }

        return complexity_mapping.get(complexity_lower, 0.5)


class ModelSelectionRequest(BaseModel):
    """
    Model selection request that contains the prompt and context information
    needed for intelligent model routing, including tool usage detection.
    """

    # The user prompt to analyze
    prompt: str

    # Tool-related fields for function calling detection
    tool_call: dict[str, Any] | None = None  # Current tool call being made
    tools: list[dict[str, Any]] | None = None  # Available tool definitions

    # Our custom parameters for model selection
    user_id: str | None = None

    models: list[ModelCapability] | None = None
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
    """Simplified response with just the selected model and alternatives."""

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
