"""
OpenAI-specific response models.
"""

from typing import List
from adaptive.models.completions.base import BaseUsage, BaseChoice, BaseProviderResponse  # type: ignore


class OpenAIUsage(BaseUsage):
    """
    OpenAI usage statistics.
    """

    pass


class OpenAIChoice(BaseChoice):
    """
    OpenAI completion choice.
    """

    pass


class OpenAIResponse(BaseProviderResponse):
    """
    OpenAI API response format.
    """

    object: str
    created: int
    choices: List[OpenAIChoice]
    usage: OpenAIUsage
