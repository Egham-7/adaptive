from functools import lru_cache
from typing import Any

from pydantic import BaseModel
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

    # Server settings - can be overridden by environment variables
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    # FastAPI settings
    workers: int = 1
    reload: bool = False
    access_log: bool = True
    log_level: str = "info"

    # Logging settings
    logging_level: str = "INFO"

    # Legacy nested config objects (kept for backward compatibility)
    server: ServerConfig = ServerConfig()
    fastapi: FastAPIConfig = FastAPIConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        # Allow environment variables to override nested settings
        env_nested_delimiter="__",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Sync individual settings with nested objects
        self.server.host = self.host
        self.server.port = self.port
        self.fastapi.workers = self.workers
        self.fastapi.reload = self.reload
        self.fastapi.access_log = self.access_log
        self.fastapi.log_level = self.log_level
        self.logging.level = self.logging_level


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
