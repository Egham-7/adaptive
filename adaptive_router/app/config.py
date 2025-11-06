"""Application configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_REGISTRY_TIMEOUT = 5.0
DEFAULT_MODEL_COST = 1.0
FUZZY_MATCH_SIMILARITY_THRESHOLD = 0.8


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = Field(default=DEFAULT_HOST, description="Server host")
    port: int = Field(default=DEFAULT_PORT, description="Server port")

    # Model Registry settings
    model_registry_base_url: str = Field(
        default="http://localhost:3000",
        description="Base URL for model registry",
    )
    model_registry_timeout: float = Field(
        default=DEFAULT_REGISTRY_TIMEOUT,
        description="Timeout for registry requests in seconds",
    )
    default_model_cost: float = Field(
        default=DEFAULT_MODEL_COST,
        description="Default cost per 1M tokens when registry pricing is missing",
    )

    # MinIO/S3 settings
    minio_private_endpoint: str = Field(
        default="http://localhost:9000",
        description="Private MinIO endpoint URL",
    )
    minio_root_user: str = Field(default="minioadmin", description="MinIO root user")
    minio_root_password: str = Field(
        default="minioadmin",
        description="MinIO root password",
    )
    s3_bucket_name: str = Field(
        default="adaptive-router-profiles",
        description="S3 bucket name",
    )
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_profile_key: str = Field(
        default="global/profile.json",
        description="S3 profile key path",
    )
    s3_connect_timeout: str = Field(default="5", description="S3 connect timeout")
    s3_read_timeout: str = Field(default="30", description="S3 read timeout")

    # CORS settings
    allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins",
    )

    @property
    def origins_list(self) -> list[str]:
        """Parse allowed origins into a list."""
        if not self.allowed_origins:
            return ["*"]
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]
