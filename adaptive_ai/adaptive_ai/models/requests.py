from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for model selection")
