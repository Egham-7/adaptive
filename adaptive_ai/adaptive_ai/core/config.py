from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000


class FastAPIConfig(BaseModel):
    """FastAPI server configuration."""

    workers: int = 1
    reload: bool = False
    access_log: bool = True
    log_level: str = "info"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


class Settings(BaseSettings):
    """Main application settings."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    fastapi: FastAPIConfig = Field(default_factory=FastAPIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

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
