"""Classification models for prompt analysis and task complexity detection."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Results from prompt classification including task type and complexity metrics.

    This model contains the output from ML classifiers that analyze prompts
    to determine task types, complexity scores, and various task characteristics.
    All list fields contain results for batch processing where each index
    corresponds to a single prompt in the batch.

    Attributes:
        task_type_1: Primary task type classifications for each prompt
        task_type_2: Secondary task type classifications for each prompt
        task_type_prob: Confidence scores for primary task types
        creativity_scope: Creativity level scores (0.0-1.0)
        reasoning: Reasoning complexity scores (0.0-1.0)
        contextual_knowledge: Required contextual knowledge scores (0.0-1.0)
        prompt_complexity_score: Overall complexity scores (0.0-1.0)
        domain_knowledge: Domain-specific knowledge requirement scores (0.0-1.0)
        number_of_few_shots: Few-shot learning requirement scores (0.0-1.0)
        no_label_reason: Confidence in classification accuracy (0.0-1.0)
        constraint_ct: Number of constraints detected in prompts (0.0-1.0)
    """

    task_type_1: list[str] = Field(
        description="Primary task type for each prompt in batch",
        examples=[["Text Generation", "Code Generation"]],
    )
    task_type_2: list[str] = Field(
        description="Secondary task type for each prompt in batch",
        examples=[["Summarization", "Classification"]],
    )
    task_type_prob: list[float] = Field(
        description="Confidence scores for primary task types", examples=[[0.89, 0.76]]
    )
    creativity_scope: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Creativity level required for each task (0=analytical, 1=creative)",
        examples=[[0.2, 0.8]],
    )
    reasoning: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Reasoning complexity required (0=simple, 1=complex)",
        examples=[[0.7, 0.4]],
    )
    contextual_knowledge: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Context knowledge requirement (0=none, 1=extensive)",
        examples=[[0.3, 0.6]],
    )
    prompt_complexity_score: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Overall prompt complexity (0=simple, 1=complex)",
        examples=[[0.45, 0.72]],
    )
    domain_knowledge: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Domain-specific knowledge requirement (0=general, 1=specialist)",
        examples=[[0.1, 0.9]],
    )
    number_of_few_shots: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Few-shot learning requirement (0=none, 1=many examples)",
        examples=[[0.0, 0.3]],
    )
    no_label_reason: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Confidence in classification accuracy (0=low, 1=high)",
        examples=[[0.9, 0.85]],
    )
    constraint_ct: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description="Constraint complexity detected (0=none, 1=many constraints)",
        examples=[[0.2, 0.5]],
    )
