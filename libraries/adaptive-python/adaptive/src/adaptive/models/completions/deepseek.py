"""
DeepSeek-specific response models.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel  # type: ignore
from adaptive.models.completions.base import BaseUsage, BaseChoice, BaseProviderResponse  # type: ignore


class DeepSeekLogProbs(BaseModel):
    """
    Log probabilities information in DeepSeek's response.
    """

    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]


class DeepSeekChoice(BaseChoice):
    """
    DeepSeek completion choice.
    """

    logprobs: Optional[DeepSeekLogProbs] = None


class DeepSeekUsage(BaseUsage):
    """
    DeepSeek usage statistics.
    """

    pass


class DeepSeekResponse(BaseProviderResponse):
    """
    DeepSeek API response format.
    """

    object: str
    created: int
    choices: List[DeepSeekChoice]
    usage: DeepSeekUsage
