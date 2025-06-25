from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: str | None = None
    max_file_size: str = "10MB"
    backup_count: int = 5


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


# NEW: Added EmbeddingCacheSettings
class EmbeddingCacheSettings(BaseModel):
    """Configuration for the embedding cache."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    similarity_threshold: float = 0.95


class Settings(BaseSettings):
    """Main application settings."""

    app: AppConfig = AppConfig()
    server: ServerConfig = ServerConfig()
    litserve: LitServeConfig = LitServeConfig()
    logging: LoggingConfig = LoggingConfig()
    metrics: MetricsConfig = MetricsConfig()
    security: SecurityConfig = SecurityConfig()
    health: HealthConfig = HealthConfig()
    # NEW: Added embedding_cache configuration
    embedding_cache: EmbeddingCacheSettings = EmbeddingCacheSettings()

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ADAPTIVE_AI_", case_sensitive=False
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
