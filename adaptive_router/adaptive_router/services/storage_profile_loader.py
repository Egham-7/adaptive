"""MinIO storage profile loader for UniRouter (Railway deployment)."""

import json
import logging
from typing import Any, Dict

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from adaptive_router.core.storage_config import MinIOSettings

logger = logging.getLogger(__name__)


class StorageProfileLoader:
    """Load UniRouter profiles from MinIO storage.

    This class downloads profile data from MinIO and returns it in the same
    format as the local JSON files. It's designed to be a drop-in replacement
    for the file-based loading mechanism.

    Performance:
        - Download time: ~100-200ms for 376KB profile
        - Cached in memory after first load (by unirouter_service.py)

    Example:
        Railway deployment:
        >>> from adaptive_router.services.storage_profile_loader import StorageProfileLoader
        >>> from adaptive_router.core.storage_config import MinIOSettings
        >>> settings = MinIOSettings()  # Reads from Railway env vars
        >>> loader = StorageProfileLoader.from_minio_settings(settings)
        >>> profile = loader.load_global_profile()
        >>> print(profile.keys())
        dict_keys(['cluster_centers', 'llm_profiles', 'tfidf_vocabulary', 'scaler_parameters', 'metadata'])
    """

    def __init__(
        self,
        bucket_name: str,
        region: str,
        profile_key: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        connect_timeout: int = 5,
        read_timeout: int = 30,
    ):
        """Initialize MinIO profile loader.

        Args:
            bucket_name: Bucket name
            region: Region (required by boto3, ignored by MinIO)
            profile_key: Key for profile (default: global/profile.json)
            endpoint_url: MinIO endpoint URL
            access_key_id: MinIO root user
            secret_access_key: MinIO root password
            connect_timeout: Connection timeout in seconds (default: 5)
            read_timeout: Read timeout in seconds (default: 30)
        """
        self.bucket_name = bucket_name
        self.profile_key = profile_key
        self.endpoint_url = endpoint_url

        # Configure S3 client with retries and s3v4 signature for MinIO
        config = Config(
            region_name=region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        # Initialize MinIO client (S3-compatible)
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            config=config,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        logger.info(
            f"StorageProfileLoader initialized: MinIO storage, bucket={bucket_name}, "
            f"key={profile_key}, endpoint={endpoint_url}, "
            f"timeouts=(connect:{connect_timeout}s, read:{read_timeout}s)"
        )

    @classmethod
    def from_minio_settings(cls, settings: MinIOSettings) -> "StorageProfileLoader":
        """Create loader from MinIOSettings (Railway native).

        Args:
            settings: MinIO configuration settings

        Returns:
            StorageProfileLoader instance configured for MinIO
        """
        return cls(
            bucket_name=settings.bucket_name,
            region=settings.region,
            profile_key=settings.profile_key,
            endpoint_url=settings.endpoint_url,
            access_key_id=settings.root_user,
            secret_access_key=settings.root_password,
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
        )

    def load_global_profile(self) -> Dict[str, Any]:
        """Load global profile from MinIO storage.

        This method downloads the profile from MinIO and returns it as a dictionary
        with the following structure:
            {
                'cluster_centers': {...},
                'llm_profiles': {...},
                'tfidf_vocabulary': {...},
                'scaler_parameters': {...},
                'metadata': {...}
            }

        Returns:
            Profile data dictionary

        Raises:
            FileNotFoundError: If profile doesn't exist in MinIO
            ValueError: If profile data is corrupted
            ClientError: If MinIO operation fails
        """
        try:
            logger.info(
                f"Loading profile from MinIO: s3://{self.bucket_name}/{self.profile_key}"
            )

            # Download from MinIO
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.profile_key)

            # Read and parse JSON
            data = response["Body"].read()
            profile = json.loads(data)

            # Log success
            size_kb = len(data) / 1024
            logger.info(
                f"Successfully loaded profile from MinIO "
                f"(size: {size_kb:.1f}KB, n_clusters: {profile.get('metadata', {}).get('n_clusters', 'N/A')})"
            )

            return profile

        except self.s3.exceptions.NoSuchKey:
            error_msg = f"Profile not found in MinIO: s3://{self.bucket_name}/{self.profile_key}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        except json.JSONDecodeError as e:
            error_msg = f"Corrupted profile data in MinIO: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = f"MinIO error loading profile: {error_code} - {e}"
            logger.error(error_msg)
            raise

    def health_check(self) -> bool:
        """Check if MinIO bucket is accessible.

        Returns:
            True if bucket exists and is accessible, False otherwise
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"MinIO bucket {self.bucket_name} is accessible")
            return True
        except ClientError as e:
            logger.warning(f"MinIO bucket {self.bucket_name} not accessible: {e}")
            return False
