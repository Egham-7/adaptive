from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


class CORSConfig(BaseModel):
    """CORS configuration."""

    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="List of allowed origins for CORS",
    )
    allow_credentials: bool = Field(
        default=True,
        description="Whether to allow credentials in CORS requests",
    )
    allow_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="List of allowed HTTP methods",
    )
    allow_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of allowed headers",
    )


class ClassifierConfig(BaseModel):
    """ML classifier configuration."""

    model_name: str = Field(
        default="nvidia/prompt-task-and-complexity-classifier",
        description="HuggingFace model identifier for the classifier",
    )
    device: str = Field(
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        description="Device to run inference on (cuda/cpu)",
    )
    max_length: int = Field(
        default=512,
        description="Maximum token length for input prompts",
    )


class Settings(BaseSettings):  # type: ignore[misc]
    """Main application settings."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        # Allow environment variables to override nested settings
        env_nested_delimiter="__",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
