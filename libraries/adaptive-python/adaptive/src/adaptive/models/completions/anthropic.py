"""
Anthropic-specific response models.
"""

from typing import List, Optional
from pydantic import BaseModel  # type: ignore
from adaptive.models.completions.base import BaseProviderResponse  # type: ignore


class AnthropicContentItem(BaseModel):
    """
    Content item in Anthropic's response.
    """

    type: str
    text: str


class AnthropicUsage(BaseModel):
    """
    Anthropic usage statistics.
    """

    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseProviderResponse):
    """
    Anthropic API response format.
    """

    type: str
    role: str
    content: List[AnthropicContentItem]
    stop_reason: str
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage
