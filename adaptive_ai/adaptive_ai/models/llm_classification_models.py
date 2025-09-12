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

    This model contains the output from ML classifiers that analyze prompts
    to determine task types, complexity scores, and various task characteristics.
    All list fields contain results for batch processing where each index
    corresponds to a single prompt in the batch.

    Attributes:
        task_type_1: Primary task type classifications for each prompt (required)
        task_type_2: Secondary task type classifications for each prompt
        task_type_prob: Confidence scores for primary task types
        creativity_scope: Creativity level scores (0.0-1.0)
        reasoning: Reasoning complexity scores (0.0-1.0)
        contextual_knowledge: Required contextual knowledge scores (0.0-1.0)
        prompt_complexity_score: Overall complexity scores (0.0-1.0) (required)
        domain_knowledge: Domain-specific knowledge requirement scores (0.0-1.0)
        number_of_few_shots: Few-shot learning requirement scores (integer values)
        no_label_reason: Confidence in classification accuracy (0.0-1.0)
        constraint_ct: Number of constraints detected in prompts (0.0-1.0)
    """

    # Required fields from classifier output
    task_type_1: list[str] = Field(
        description="Primary task type for each prompt in batch (required)",
        examples=[["Text Generation", "Code Generation"]],
    )
    prompt_complexity_score: list[UnitFloat] = Field(
        description="Overall prompt complexity (0=simple, 1=complex) (required)",
        examples=[[0.45, 0.72]],
    )

    # Optional detailed fields
    task_type_2: list[str] | None = Field(
        default=None,
        description="Secondary task type for each prompt in batch",
        examples=[["Summarization", "Classification"]],
    )
    task_type_prob: list[UnitFloat] | None = Field(
        default=None,
        description="Confidence scores for primary task types",
        examples=[[0.89, 0.76]],
    )
    creativity_scope: list[UnitFloat] | None = Field(
        default=None,
        description="Creativity level required for each task (0=analytical, 1=creative)",
        examples=[[0.2, 0.8]],
    )
    reasoning: list[UnitFloat] | None = Field(
        default=None,
        description="Reasoning complexity required (0=simple, 1=complex)",
        examples=[[0.7, 0.4]],
    )
    contextual_knowledge: list[UnitFloat] | None = Field(
        default=None,
        description="Context knowledge requirement (0=none, 1=extensive)",
        examples=[[0.3, 0.6]],
    )
    domain_knowledge: list[UnitFloat] | None = Field(
        default=None,
        description="Domain-specific knowledge requirement (0=general, 1=specialist)",
        examples=[[0.1, 0.9]],
    )
    number_of_few_shots: list[int] | None = Field(
        default=None,
        description="Few-shot learning requirement (number of examples needed)",
        examples=[[0, 3]],
    )
    no_label_reason: list[UnitFloat] | None = Field(
        default=None,
        description="Confidence in classification accuracy (0=low, 1=high)",
        examples=[[0.9, 0.85]],
    )
    constraint_ct: list[UnitFloat] | None = Field(
        default=None,
        description="Constraint complexity detected (0=none, 1=many constraints)",
        examples=[[0.2, 0.5]],
    )


class ClassifyRequest(BaseModel):
    """Request model for batch prompt classification API."""

    prompts: list[PromptsItem] = Field(
        description="List of prompts to classify", min_length=1, max_length=100
    )


class SingleClassifyRequest(BaseModel):
    """Request model for single prompt classification API."""

    prompt: PromptsItem = Field(description="Single prompt to classify")
