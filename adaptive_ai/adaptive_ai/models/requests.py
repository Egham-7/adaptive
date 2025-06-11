from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from services.llm_parameters import OpenAIParameters


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for model selection")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ModelSelectionResponse(BaseModel):
    selected_model: str = Field(..., description="Name of the selected model")
    confidence: float = Field(..., description="Confidence score for the selection")
    reasoning: Optional[str] = Field(
        None, description="Explanation for the model selection"
    )
    alternatives: Optional[List[str]] = Field(
        None, description="Alternative model suggestions"
    )
    parameters: OpenAIParameters = Field(
        ..., description="OpenAI parameters for the selected model"
    )
    provider: Optional[str] = Field(None, description="Model provider")
