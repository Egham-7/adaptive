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

    Note: All fields are required to match the prompt-task-complexity-classifier service output.

    Attributes:
        task_type_1: Primary task type classification (required)
        task_type_2: Secondary task type classification (required)
        task_type_prob: Confidence score for primary task type (required)
        creativity_scope: Creativity level score (0.0-1.0) (required)
        reasoning: Reasoning complexity score (0.0-1.0) (required)
        contextual_knowledge: Required contextual knowledge score (0.0-1.0) (required)
        prompt_complexity_score: Overall complexity score (0.0-1.0) (required)
        domain_knowledge: Domain-specific knowledge requirement score (0.0-1.0) (required)
        number_of_few_shots: Few-shot learning requirement score (required)
        no_label_reason: Confidence in classification accuracy (0.0-1.0) (required)
        constraint_ct: Number of constraints detected in prompt (0.0-1.0) (required)
    """

    # All fields are required to match prompt-task-complexity-classifier output
    task_type_1: str = Field(
        description="Primary task type (required)",
        examples=["Text Generation", "Code Generation"],
    )
    task_type_2: str = Field(
        description="Secondary task type (required)",
        examples=["Summarization", "Classification"],
    )
    task_type_prob: UnitFloat = Field(
        description="Confidence score for primary task type (required)",
        examples=[0.89, 0.76],
    )
    creativity_scope: UnitFloat = Field(
        description="Creativity level required for task (0=analytical, 1=creative) (required)",
        examples=[0.2, 0.8],
    )
    reasoning: UnitFloat = Field(
        description="Reasoning complexity required (0=simple, 1=complex) (required)",
        examples=[0.7, 0.4],
    )
    contextual_knowledge: UnitFloat = Field(
        description="Context knowledge requirement (0=none, 1=extensive) (required)",
        examples=[0.3, 0.6],
    )
    prompt_complexity_score: UnitFloat = Field(
        description="Overall prompt complexity (0=simple, 1=complex) (required)",
        examples=[0.45, 0.72],
    )
    domain_knowledge: UnitFloat = Field(
        description="Domain-specific knowledge requirement (0=general, 1=specialist) (required)",
        examples=[0.1, 0.9],
    )
    number_of_few_shots: float = Field(
        description="Few-shot learning requirement score (required)",
        examples=[0.0, 0.3],
    )
    no_label_reason: UnitFloat = Field(
        description="Confidence in classification accuracy (0=low, 1=high) (required)",
        examples=[0.9, 0.85],
    )
    constraint_ct: UnitFloat = Field(
        description="Constraint complexity detected (0=none, 1=many constraints) (required)",
        examples=[0.2, 0.5],
    )


class ClassifyRequest(BaseModel):
    """Request model for single prompt classification API.

    Matches prompt-task-complexity-classifier's ClassifyRequest model.
    """

    prompt: PromptsItem = Field(description="Single prompt to classify")


class ClassifyBatchRequest(BaseModel):
    """Request model for batch prompt classification API.

    Matches prompt-task-complexity-classifier's ClassifyBatchRequest model.
    """

    prompts: list[PromptsItem] = Field(
        description="List of prompts to classify", min_length=1, max_length=100
    )
