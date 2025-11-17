"""Application configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants
DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
FUZZY_MATCH_SIMILARITY_THRESHOLD = 0.8


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MinIO/S3 settings
    minio_private_endpoint: str | None = Field(
        default=None,
        description="Private MinIO endpoint URL",
    )
    minio_public_endpoint: str | None = Field(
        default=None,
        description="Public MinIO endpoint URL",
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

    @property
    def minio_endpoint(self) -> str:
        """Return the preferred MinIO endpoint, falling back from private to public."""
        if self.minio_private_endpoint and self.minio_private_endpoint.strip():
            return self.minio_private_endpoint.strip()
        if self.minio_public_endpoint and self.minio_public_endpoint.strip():
            return self.minio_public_endpoint.strip()
        return DEFAULT_MINIO_ENDPOINT
