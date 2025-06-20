from typing import Any

from pydantic import BaseModel, Field

from adaptive_ai.models.parameters import OpenAIParameters


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for model selection")
    context: dict[str, Any] | None = Field(None, description="Additional context")
    cost_bias: float = Field(
        0.5,
        description="Cost/quality trade-off bias (0.0-1.0). 0.5 is neutral, <0.5 favors cheaper models, >0.5 favors more expensive models",
        ge=0.0,
        le=1.0,
    )


class ModelSelectionResponse(BaseModel):
    selected_model: str = Field(..., description="Name of the selected model")
    confidence: float = Field(..., description="Confidence score for the selection")
    reasoning: str | None = Field(
        None, description="Explanation for the model selection"
    )
    alternatives: list[str] | None = Field(
        None, description="Alternative model suggestions"
    )
    parameters: OpenAIParameters = Field(
        ..., description="OpenAI parameters for the selected model"
    )
    provider: str | None = Field(None, description="Model provider")
