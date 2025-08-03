"""
Unified model representation for both registry and custom models.
Clean, readable, and efficient data structure.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Model:
    """A unified model representation for both registry and custom models."""

    # Identity
    provider: str  # "openai", "anthropic",
    name: str  # "gpt-4o", "kendrick-chat"

    # Economics
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float

    # Capabilities
    max_context_tokens: int
    supports_function_calling: bool

    # Task fitness
    best_for_task: str | None = None  # "OPEN_QA", "CODE_GENERATION", etc.
    complexity_level: float = 0.5  # 0.0 (simple) to 1.0 (complex)

    def can_handle_prompt_size(self, token_count: int) -> bool:
        """Check if model can handle the prompt length."""
        return self.max_context_tokens >= token_count

    def estimate_cost(self, input_tokens: int, output_tokens: int = 100) -> float:
        """Calculate estimated cost in dollars."""
        input_cost = input_tokens * self.cost_per_million_input_tokens / 1_000_000
        output_cost = output_tokens * self.cost_per_million_output_tokens / 1_000_000
        return input_cost + output_cost

    @property
    def is_budget_friendly(self) -> bool:
        """True if this is a cost-effective model."""
        return self.cost_per_million_input_tokens < 1.0

    @property
    def unique_id(self) -> str:
        """Unique identifier for this model."""
        return f"{self.provider}:{self.name}"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.provider}/{self.name} (${self.cost_per_million_input_tokens:.2f}/1M tokens)"


# Complexity level constants for readability
class ComplexityLevel:
    """Standard complexity levels for model capabilities."""

    SIMPLE = 0.1  # Basic tasks, simple Q&A
    MODERATE = 0.5  # Standard tasks, some reasoning
    ADVANCED = 0.8  # Complex tasks, deep reasoning

    @staticmethod
    def from_string(complexity_string: str) -> float:
        """Convert human-readable complexity to numerical score."""
        complexity_levels = {
            "easy": ComplexityLevel.SIMPLE,
            "medium": ComplexityLevel.MODERATE,
            "hard": ComplexityLevel.ADVANCED,
        }

        normalized = complexity_string.lower().strip()
        return complexity_levels.get(
            normalized, ComplexityLevel.MODERATE
        )  # Default to medium


def create_model_from_capability(capability: Any) -> Model:
    """Convert a ModelCapability to a unified Model."""
    # Handle complexity conversion
    if hasattr(capability, "complexity") and capability.complexity:
        complexity_score = ComplexityLevel.from_string(capability.complexity)
    else:
        complexity_score = ComplexityLevel.MODERATE

    return Model(
        provider=str(capability.provider),
        name=capability.model_name,
        cost_per_million_input_tokens=capability.cost_per_1m_input_tokens or 0.0,
        cost_per_million_output_tokens=capability.cost_per_1m_output_tokens or 0.0,
        max_context_tokens=capability.max_context_tokens or 4096,
        supports_function_calling=capability.supports_function_calling or False,
        best_for_task=getattr(capability, "task_type", None),
        complexity_level=complexity_score,
    )


def create_model_from_dict(model_spec: dict[str, Any]) -> Model:
    """Convert a dictionary specification to a unified Model."""
    complexity_score = ComplexityLevel.from_string(
        model_spec.get("complexity", "medium")
    )

    return Model(
        provider=model_spec["provider"],
        name=model_spec["model_name"],
        cost_per_million_input_tokens=model_spec["cost_per_1m_input_tokens"],
        cost_per_million_output_tokens=model_spec["cost_per_1m_output_tokens"],
        max_context_tokens=model_spec["max_context_tokens"],
        supports_function_calling=model_spec["supports_function_calling"],
        best_for_task=model_spec.get("task_type"),
        complexity_level=complexity_score,
    )
