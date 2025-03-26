from typing import Any, Union
from provider.provider import OpenAIResponse, GroqResponse, DeepSeekResponse

def parse_provider_response(provider: str, response: dict) -> Union[OpenAIResponse, GroqResponse, DeepSeekResponse]:
    """Parses API response based on provider type."""
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIResponse(**response)
    elif provider == "groq":
        return GroqResponse(**response)
    elif provider == "deepseek":
        return DeepSeekResponse(**response)
    
    raise ValueError("Must be a valid provider.")
