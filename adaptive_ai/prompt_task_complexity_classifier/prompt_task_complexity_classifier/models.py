"""Pydantic models for API requests and responses."""

from typing import List, Annotated
from pydantic import BaseModel, Field, conlist


class ClassificationResult(BaseModel):
    """Classification result for a single prompt."""

    task_type_1: str = Field(description="Primary task type")
    task_type_2: str = Field(description="Secondary task type")
    task_type_prob: float = Field(
        description="Confidence score for primary task type", ge=0, le=1
    )
    creativity_scope: float = Field(
        description="Creativity level required (0-1)", ge=0, le=1
    )
    reasoning: float = Field(
        description="Reasoning complexity required (0-1)", ge=0, le=1
    )
    contextual_knowledge: float = Field(
        description="Context knowledge requirement (0-1)", ge=0, le=1
    )
    prompt_complexity_score: float = Field(
        description="Overall prompt complexity (0-1)", ge=0, le=1
    )
    domain_knowledge: float = Field(
        description="Domain-specific knowledge requirement (0-1)", ge=0, le=1
    )
    number_of_few_shots: float = Field(description="Few-shot learning requirement")
    no_label_reason: float = Field(
        description="Confidence in classification accuracy (0-1)", ge=0, le=1
    )
    constraint_ct: float = Field(
        description="Constraint complexity detected (0-1)", ge=0, le=1
    )


class ClassifyRequest(BaseModel):
    """Request model for prompt classification."""

    prompt: str = Field(
        description="Prompt to classify", min_length=1, max_length=10000
    )


class ClassifyBatchRequest(BaseModel):
    """Request model for batch prompt classification."""

    prompts: Annotated[List[str], conlist(str, min_length=1, max_length=100)] = Field(
        description="List of prompts to classify"
    )
