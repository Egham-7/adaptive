"""Configuration management for the supervisor agent system."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration."""

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.1, description="Temperature for LLM responses")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for responses")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the application configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def validate_config() -> bool:
    """Validate that all required configuration is present."""
    try:
        config = get_config()
        if not config.openai_api_key:
            return False
        return True
    except Exception:
        return False


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment or config."""
    config = get_config()
    if config.openai_api_key:
        return config.openai_api_key
    
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or create a .env file with OPENAI_API_KEY=your_key_here"
        )
    return api_key