from typing import Dict, Any, TypedDict, Literal
from pydantic import BaseModel, Field
import os
from pathlib import Path

class ModelConfig(BaseModel):
    """Configuration for model capabilities and parameters"""
    description: str
    provider: Literal["GROQ", "OpenAI", "DEEPSEEK", "Anthropic"]
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)

class TaskConfig(BaseModel):
    """Configuration for task-specific parameters"""
    easy: Dict[str, Any]
    medium: Dict[str, Any]
    hard: Dict[str, Any]

class DomainConfig(BaseModel):
    """Configuration for domain-specific weights"""
    weights: list[float]
    description: str

class AppConfig(BaseModel):
    """Main application configuration"""
    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Model settings
    default_model: str = Field(default="gpt-4-turbo")
    fallback_model: str = Field(default="gpt-3.5-turbo")
    
    # Cache settings
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)  # 1 hour
    
    # Logging settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_period: int = Field(default=60)  # 1 minute

def load_config() -> AppConfig:
    """Load configuration from environment variables or config file"""
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    
    if os.path.exists(config_path):
        # Load from YAML file
        import yaml
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    else:
        # Load from environment variables
        config_data = {
            "api_host": os.getenv("API_HOST", "0.0.0.0"),
            "api_port": int(os.getenv("API_PORT", "8000")),
            "api_workers": int(os.getenv("API_WORKERS", "4")),
            "default_model": os.getenv("DEFAULT_MODEL", "gpt-4-turbo"),
            "fallback_model": os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo"),
            "cache_size": int(os.getenv("CACHE_SIZE", "1000")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            "rate_limit_period": int(os.getenv("RATE_LIMIT_PERIOD", "60")),
        }
    
    return AppConfig(**config_data)

# Load configuration
config = load_config()

# Export configuration
__all__ = ["config", "AppConfig", "ModelConfig", "TaskConfig", "DomainConfig"]
