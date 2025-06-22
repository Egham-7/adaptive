from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Input prompt for model selection",
        min_length=1,
        max_length=4096,
    )
