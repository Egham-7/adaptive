"""MinIO storage configuration for Railway deployment."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MinIOSettings(BaseSettings):
    """MinIO storage configuration (Railway native).

    This class uses Railway's MinIO template environment variables directly.

    Environment variables:
        MINIO_PUBLIC_ENDPOINT: MinIO public endpoint URL (required)
        MINIO_ROOT_USER: MinIO root username (required)
        MINIO_ROOT_PASSWORD: MinIO root password (required)
        S3_BUCKET_NAME: Bucket name (required)
        S3_REGION: Region (default: us-east-1, ignored by MinIO but required by boto3)
        S3_PROFILE_KEY: Key for profile (default: global/profile.json)
        S3_CONNECT_TIMEOUT: Connection timeout in seconds (default: 5)
        S3_READ_TIMEOUT: Read timeout in seconds (default: 30)

    Example:
        Railway deployment:
        >>> from adaptive_router.core.storage_config import MinIOSettings
        >>> settings = MinIOSettings()  # Reads from Railway env vars
        >>> print(settings.endpoint_url)
        https://minio-production-xxxx.up.railway.app

        Local development:
        >>> settings = MinIOSettings(
        ...     endpoint_url="http://localhost:9000",
        ...     root_user="minioadmin",
        ...     root_password="minioadmin",
        ...     bucket_name="adaptive-router-profiles"
        ... )
    """

    endpoint_url: str = Field(alias="MINIO_PUBLIC_ENDPOINT")
    root_user: str = Field(alias="MINIO_ROOT_USER")
    root_password: str = Field(alias="MINIO_ROOT_PASSWORD")

    bucket_name: str = Field(alias="S3_BUCKET_NAME")
    region: str = Field(default="us-east-1", alias="S3_REGION")
    profile_key: str = Field(default="global/profile.json", alias="S3_PROFILE_KEY")

    # Timeout configuration (configurable for different network conditions)
    connect_timeout: int = Field(default=5, alias="S3_CONNECT_TIMEOUT")
    read_timeout: int = Field(default=30, alias="S3_READ_TIMEOUT")

    model_config = SettingsConfigDict(
        populate_by_name=True,
        case_sensitive=False,
    )

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate that endpoint_url is a valid URL.

        Args:
            v: The endpoint URL to validate

        Returns:
            The validated URL

        Raises:
            ValueError: If URL doesn't start with http:// or https://
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"endpoint_url must start with http:// or https://, got: {v}"
            )
        return v

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate that bucket_name is not empty.

        Args:
            v: The bucket name to validate

        Returns:
            The validated bucket name

        Raises:
            ValueError: If bucket name is empty
        """
        if not v or not v.strip():
            raise ValueError("bucket_name cannot be empty")
        return v.strip()
