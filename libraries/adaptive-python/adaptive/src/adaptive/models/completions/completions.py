"""Unified response models for chat completions."""

from typing import Union, Optional, Dict, Any, List, Literal
from functools import reduce
from pydantic import BaseModel, Field  # type: ignore
from adaptive.models.completions.openai import OpenAIResponse  # type: ignore
from adaptive.models.completions.anthropic import AnthropicResponse  # type: ignore
from adaptive.models.completions.groq import GroqResponse  # type: ignore
from adaptive.models.completions.deepseek import DeepSeekResponse  # type: ignore

ProviderResponse = Union[
    OpenAIResponse, AnthropicResponse, GroqResponse, DeepSeekResponse
]


class ChatCompletionResponse(BaseModel):
    """
    Unified response wrapper for chat completions.
    """

    provider: str
    response: ProviderResponse
    error: Optional[str] = None

    @staticmethod
    def parse_provider_response(
        provider: str, response_data: Dict[str, Any]
    ) -> ProviderResponse:
        """
        Parse the response from a provider into the appropriate response model.

        Args:
            provider: The name of the provider (e.g., 'openai', 'anthropic')
            response_data: The raw response data from the provider

        Returns:
            A provider-specific response object
        """
        if not response_data:
            raise ValueError("Response data is empty")

        if provider == "openai":
            return OpenAIResponse(**response_data)
        elif provider == "anthropic":
            return AnthropicResponse(**response_data)
        elif provider == "groq":
            return GroqResponse(**response_data)
        elif provider == "deepseek":
            return DeepSeekResponse(**response_data)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class ChatCompletionStreamingResponse(BaseModel):
    """
    Unified model for streaming responses from any provider.
    """

    id: str
    model: str
    provider: Literal["openai", "groq", "deepseek", "anthropic"]
    choices: List[Dict[str, Any]] = Field(
        ..., description="The streaming completion choices"
    )
    object: Optional[str] = None
    created: Optional[int] = None
    # Provider-specific extraction paths
    _EXTRACTION_PATHS = {
        "openai": [["delta", "content"]],
        "groq": [["delta", "content"]],
        "deepseek": [["text"], ["message", "content"]],
        "anthropic": [["content"]],
    }
    # Fallback paths to try for all providers
    _FALLBACK_PATHS = [
        ["delta", "content"],
        ["text"],
        ["content"],
        ["message", "content"],
    ]

    def extract_content(self) -> str:
        """
        Extracts the text content from a streaming response.
        Returns:
            str: The extracted content or an empty string if no content is found.
        """
        if not self.choices:
            return ""
        choice = self.choices[0]
        # Try provider-specific paths first
        provider_paths = self._EXTRACTION_PATHS.get(self.provider, [])
        for path in provider_paths:
            content = self._get_nested_value(choice, path)
            if content:
                return content
        # Fall back to common paths
        for path in self._FALLBACK_PATHS:
            content = self._get_nested_value(choice, path)
            if content:
                return content
        return ""

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: List[str]) -> str:
        """
        Get a value from a nested dictionary using a path of keys.
        Args:
            data: Dictionary to extract value from
            path: List of keys forming the path to the value
        Returns:
            The value at the specified path or empty string if not found
        """
        try:
            # More idiomatic way to traverse nested dictionaries
            return reduce(lambda d, key: d.get(key, {}), path[:-1], data).get(
                path[-1], ""
            )
        except (AttributeError, TypeError):
            # Handle case where an intermediate value is not a dictionary
            return ""
