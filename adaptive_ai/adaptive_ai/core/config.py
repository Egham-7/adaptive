from functools import lru_cache
import os
from pathlib import Path
import re

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "::"
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
    """Main application settings with YAML config support."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    litserve: LitServeConfig = Field(default_factory=LitServeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ADAPTIVE_AI_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        """Load settings from YAML file with environment variable substitution."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            yaml_content = f.read()

        # Substitute environment variables
        yaml_content = cls._substitute_env_vars(yaml_content)

        # Parse YAML and create settings
        config_data = yaml.safe_load(yaml_content)
        return cls(**config_data)

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Substitute environment variables in YAML content."""
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_expr = match.group(1)
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                return os.getenv(var_expr.strip(), f"${{{var_expr}}}")

        return re.sub(pattern, replace_var, content)


@lru_cache
def get_settings(config_file: str | None = None) -> Settings:
    """Get cached settings instance."""
    config_file = config_file or os.getenv("ADAPTIVE_AI_CONFIG_FILE")

    if config_file:
        return Settings.from_yaml(config_file)

    return Settings()
