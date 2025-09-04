from typing import Any

from pydantic import BaseModel, Field, model_validator

from .llm_enums import TaskType

# =============================================================================
# Model Capability & Configuration Models
# =============================================================================


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
    task_type: TaskType | str | None = None
    complexity: str | None = None  # "easy", "medium", "hard"

    # Metadata
    description: str | None = None
    languages_supported: list[str] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "ModelCapability":
        """Ensure at least one field is present in the model capability."""
        fields_to_check = [
            self.provider,
            self.model_name,
            self.cost_per_1m_input_tokens,
            self.cost_per_1m_output_tokens,
            self.max_context_tokens,
            self.supports_function_calling,
            self.task_type,
            self.complexity,
            self.description,
        ]

        # Check if languages_supported has any items
        has_languages = bool(self.languages_supported)

        # Check if any field has a non-None value
        has_non_none_field = any(field is not None for field in fields_to_check)

        if not has_non_none_field and not has_languages:
            raise ValueError(
                "ModelCapability must have at least one non-None field specified"
            )

        return self


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


class Alternative(BaseModel):
    provider: str
    model: str


class ModelSelectionResponse(BaseModel):
    """Simplified response with just the selected model and alternatives."""

    provider: str
    model: str
    alternatives: list[Alternative]
