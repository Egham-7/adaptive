"""
Groq-specific response models.
"""

from typing import List
from adaptive.models.completions.base import BaseUsage, BaseChoice, BaseProviderResponse  # type: ignore


class GroqUsage(BaseUsage):
    """
    Groq usage statistics.
    """

    pass


class GroqChoice(BaseChoice):
    """
    Groq completion choice.
    """

    pass


class GroqResponse(BaseProviderResponse):
    """
    Groq API response format.
    """

    created: int
    choices: List[GroqChoice]
    usage: GroqUsage
