from functools import lru_cache
import os
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


class AppConfig(BaseModel):
    """Application-level configuration."""

    name: str = "adaptive-ai"
    version: str = "0.1.0"
    description: str = "Intelligent LLM Infrastructure with Smart Model Selection"
    environment: str = "development"
    debug: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    max_requests: int = 1000
    max_requests_jitter: int = 50


class LitServeConfig(BaseModel):
    """LitServe configuration."""

    accelerator: str = "auto"
    devices: str = "auto"
    max_batch_size: int = 8
    batch_timeout: float = 0.05
    workers: int = 1
    timeout: float = 30.0


class ModelSelectionConfig(BaseModel):
    """Model selection configuration."""

    default_model: str = "gpt-3.5-turbo"
    threshold: float = 0.7
    fallback_model: str = "gpt-3.5-turbo"
    cache_embeddings: bool = True
    cache_ttl: int = 3600
    cost_bias: float = 0.5  # Default to 0.5 (no bias), range 0.0-1.0


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: str | None = None
    max_file_size: str = "10MB"
    backup_count: int = 5


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    backend: str = "memory"
    ttl: int = 3600
    max_size: int = 1000
    redis: RedisConfig = RedisConfig()


class PrometheusConfig(BaseModel):
    """Prometheus configuration."""

    enabled: bool = True
    port: int = 9090


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    enabled: bool = True
    endpoint: str = "/metrics"
    include_model_metrics: bool = True
    include_performance_metrics: bool = True
    prometheus: PrometheusConfig = PrometheusConfig()


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10


class CorsConfig(BaseModel):
    """CORS configuration."""

    enabled: bool = True
    origins: list[str] = ["*"]
    methods: list[str] = ["GET", "POST", "OPTIONS"]
    headers: list[str] = ["*"]


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key_required: bool = False
    rate_limiting: RateLimitingConfig = RateLimitingConfig()
    cors: CorsConfig = CorsConfig()


class HealthConfig(BaseModel):
    """Health check configuration."""

    endpoint: str = "/health"
    check_models: bool = True
    check_dependencies: bool = True
    timeout: float = 5.0


class ProviderConfig(BaseModel):
    """Provider API configuration."""

    base_url: str
    timeout: int = 30
    max_retries: int = 3


class Settings(BaseSettings):
    """Main application settings."""

    # Configuration sections
    app: AppConfig = AppConfig()
    server: ServerConfig = ServerConfig()
    litserve: LitServeConfig = LitServeConfig()
    model_selection: ModelSelectionConfig = ModelSelectionConfig()
    logging: LoggingConfig = LoggingConfig()
    cache: CacheConfig = CacheConfig()
    metrics: MetricsConfig = MetricsConfig()
    security: SecurityConfig = SecurityConfig()
    health: HealthConfig = HealthConfig()

    # File paths
    config_file: str = Field("config/config.yaml")

    class Config:
        env_file = ".env"
        env_prefix = "ADAPTIVE_AI_"
        case_sensitive = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load configuration from YAML file
        self._load_yaml_config()

    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self.get_config_file_path()
        if config_path and config_path.exists():
            yaml_config = _load_yaml_config(str(config_path))
            self._merge_yaml_config(yaml_config)

    def _merge_yaml_config(self, yaml_config: dict[str, Any]) -> None:
        """Merge YAML configuration into settings."""
        config_mapping = {
            "app": (AppConfig, "app"),
            "server": (ServerConfig, "server"),
            "litserve": (LitServeConfig, "litserve"),
            "model_selection": (ModelSelectionConfig, "model_selection"),
            "logging": (LoggingConfig, "logging"),
            "cache": (CacheConfig, "cache"),
            "metrics": (MetricsConfig, "metrics"),
            "security": (SecurityConfig, "security"),
            "health": (HealthConfig, "health"),
        }

        for key, (config_class, attr_name) in config_mapping.items():
            if key in yaml_config:
                setattr(self, attr_name, config_class(**yaml_config[key]))

    def get_config_file_path(self) -> Path | None:
        """Get the path to the configuration file."""
        if os.path.isabs(self.config_file):
            return Path(self.config_file)

        # Check if we're in a Docker container
        if os.path.exists("/app"):
            config_path = Path("/app") / self.config_file
            if config_path.exists():
                return config_path

        # Local development - get project root
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / self.config_file
        if config_path.exists():
            return config_path

        # Fallback: look in current working directory
        config_path = Path.cwd() / self.config_file
        if config_path.exists():
            return config_path

        return None

    def load_model_config(self) -> dict[str, Any]:
        """Load the complete model configuration from YAML file."""
        config_path = self.get_config_file_path()
        if config_path and config_path.exists():
            return _load_yaml_config(str(config_path))
        return {}

    def get_model_capabilities(self) -> dict[str, Any]:
        """Get model capabilities from configuration."""
        config = self.load_model_config()
        return cast(dict[str, Any], config.get("model_capabilities", {}))

    def get_task_model_mappings(self) -> dict[str, Any]:
        """Get task to model mappings from configuration."""
        config = self.load_model_config()
        return cast(dict[str, Any], config.get("task_model_mappings", {}))

    def get_task_parameters(self) -> dict[str, Any]:
        """Get task parameters from configuration."""
        config = self.load_model_config()
        return cast(dict[str, Any], config.get("task_parameters", {}))


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache(maxsize=1)
def _load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load the configuration YAML file (cached)."""
    try:
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        ) from None
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration YAML: {e}") from e
