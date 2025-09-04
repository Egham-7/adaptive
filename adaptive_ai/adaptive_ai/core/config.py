from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000


class LitServeConfig(BaseModel):
    """LitServe configuration."""

    accelerator: str = "auto"
    devices: str = "auto"
    max_batch_size: int = 8
    batch_timeout: float = 0.05


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"


class Settings(BaseSettings):
    """Main application settings."""

    server: ServerConfig = ServerConfig()
    litserve: LitServeConfig = LitServeConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ADAPTIVE_AI_", case_sensitive=False
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
