"""
Base model classes for provider responses.
"""

from pydantic import BaseModel  # type: ignore
from adaptive.models.completions.types import Message  # type: ignore


class BaseUsage(BaseModel):
    """
    Base class for usage statistics across providers.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BaseChoice(BaseModel):
    """
    Base class for completion choices across providers.
    """

    index: int
    message: Message
    finish_reason: (
        str  # "stop" | "length" | "function_call" | "content_filter" | "tool_calls"
    )


class BaseProviderResponse(BaseModel):
    """
    Base class for provider responses.
    """

    id: str
    model: str
