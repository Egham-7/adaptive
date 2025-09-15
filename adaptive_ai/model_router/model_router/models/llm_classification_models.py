"""Classification models for prompt analysis and task complexity detection."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Type alias for probability values constrained to [0.0, 1.0] range
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]

# Type alias for individual prompt strings with constraints
PromptsItem = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)
]


class ClassificationResult(BaseModel):
    """Results from prompt classification including task type and complexity metrics.

    This model contains the output from ML classifiers that analyze a single prompt
    to determine task types, complexity scores, and various task characteristics.
    For batch processing, use List[ClassificationResult].

    Attributes:
        task_type_1: Primary task type classification (required)
        task_type_2: Secondary task type classification
        task_type_prob: Confidence score for primary task type
        creativity_scope: Creativity level score (0.0-1.0)
        reasoning: Reasoning complexity score (0.0-1.0)
        contextual_knowledge: Required contextual knowledge score (0.0-1.0)
        prompt_complexity_score: Overall complexity score (0.0-1.0) (required)
        domain_knowledge: Domain-specific knowledge requirement score (0.0-1.0)
        number_of_few_shots: Few-shot learning requirement score (float value)
        no_label_reason: Confidence in classification accuracy (0.0-1.0)
        constraint_ct: Number of constraints detected in prompt (0.0-1.0)
    """

    # Required fields from classifier output
    task_type_1: str = Field(
        description="Primary task type (required)",
        examples=["Text Generation", "Code Generation"],
    )
    prompt_complexity_score: UnitFloat = Field(
        description="Overall prompt complexity (0=simple, 1=complex) (required)",
        examples=[0.45, 0.72],
    )

    # Optional detailed fields
    task_type_2: str | None = Field(
        default=None,
        description="Secondary task type",
        examples=["Summarization", "Classification"],
    )
    task_type_prob: UnitFloat | None = Field(
        default=None,
        description="Confidence score for primary task type",
        examples=[0.89, 0.76],
    )
    creativity_scope: UnitFloat | None = Field(
        default=None,
        description="Creativity level required for task (0=analytical, 1=creative)",
        examples=[0.2, 0.8],
    )
    reasoning: UnitFloat | None = Field(
        default=None,
        description="Reasoning complexity required (0=simple, 1=complex)",
        examples=[0.7, 0.4],
    )
    contextual_knowledge: UnitFloat | None = Field(
        default=None,
        description="Context knowledge requirement (0=none, 1=extensive)",
        examples=[0.3, 0.6],
    )
    domain_knowledge: UnitFloat | None = Field(
        default=None,
        description="Domain-specific knowledge requirement (0=general, 1=specialist)",
        examples=[0.1, 0.9],
    )
    number_of_few_shots: float | None = Field(
        default=None,
        description="Few-shot learning requirement score",
        examples=[0.0, 0.3],
    )
    no_label_reason: UnitFloat | None = Field(
        default=None,
        description="Confidence in classification accuracy (0=low, 1=high)",
        examples=[0.9, 0.85],
    )
    constraint_ct: UnitFloat | None = Field(
        default=None,
        description="Constraint complexity detected (0=none, 1=many constraints)",
        examples=[0.2, 0.5],
    )


class ClassifyRequest(BaseModel):
    """Request model for batch prompt classification API."""

    prompts: list[PromptsItem] = Field(
        description="List of prompts to classify", min_length=1, max_length=100
    )


class SingleClassifyRequest(BaseModel):
    """Request model for single prompt classification API."""

    prompt: PromptsItem = Field(description="Single prompt to classify")
