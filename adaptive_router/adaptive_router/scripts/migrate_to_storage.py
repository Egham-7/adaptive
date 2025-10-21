#!/usr/bin/env python3
"""Migrate local JSON files to MinIO storage (Railway deployment).

This script reads the current UniRouter JSON files from the local filesystem
and uploads them to MinIO as a single combined profile blob.

IMPORTANT: This script is READ-ONLY for local files - it does NOT delete or
modify any existing JSON files. They will remain as backups.

Usage:
    # Set environment variables:
    export S3_BUCKET_NAME=adaptive-router-profiles
    export MINIO_PUBLIC_ENDPOINT=https://minio-production.railway.app
    export MINIO_ROOT_USER=your-user
    export MINIO_ROOT_PASSWORD=your-password

    # Run migration:
    uv run python adaptive_router/scripts/migrate_to_storage.py
"""

import json
import logging
import sys
from pathlib import Path

import boto3  # type: ignore[import-untyped]
from botocore.client import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from adaptive_router.core.storage_config import MinIOSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_local_profile(data_dir: Path) -> dict:
    """Load profile data from local JSON files (READ ONLY).

    Args:
        data_dir: Directory containing cluster JSON files

    Returns:
        Combined profile data dictionary

    Raises:
        FileNotFoundError: If required files are missing
    """
    logger.info(f"Loading local JSON files from {data_dir}")

    required_files = [
        "cluster_centers.json",
        "llm_profiles.json",
        "tfidf_vocabulary.json",
        "scaler_parameters.json",
        "metadata.json",
    ]

    # Verify all files exist
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load all files (READ ONLY - no deletion or modification)
    profile_data = {
        "cluster_centers": json.load(open(data_dir / "cluster_centers.json")),
        "llm_profiles": json.load(open(data_dir / "llm_profiles.json")),
        "tfidf_vocabulary": json.load(open(data_dir / "tfidf_vocabulary.json")),
        "scaler_parameters": json.load(open(data_dir / "scaler_parameters.json")),
        "metadata": json.load(open(data_dir / "metadata.json")),
    }

    # Log summary
    n_clusters = profile_data["metadata"]["n_clusters"]
    n_models = len(profile_data["llm_profiles"])
    logger.info(f"Loaded profile: {n_clusters} clusters, {n_models} models")

    return profile_data


def upload_to_minio(profile_data: dict, settings: MinIOSettings) -> str:
    """Upload profile data to MinIO storage.

    Args:
        profile_data: Combined profile data
        settings: MinIO configuration

    Returns:
        Version ID (if versioning enabled)

    Raises:
        ClientError: If upload fails
    """
    logger.info(
        f"Uploading to MinIO: s3://{settings.bucket_name}/{settings.profile_key}"
    )
    logger.info(f"MinIO endpoint: {settings.endpoint_url}")

    # Configure S3 client with s3v4 signature for MinIO
    config = Config(
        region_name=settings.region,
        signature_version="s3v4",
    )

    # Initialize MinIO client (S3-compatible)
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        config=config,
        aws_access_key_id=settings.root_user,
        aws_secret_access_key=settings.root_password,
    )

    # Serialize to JSON
    json_data = json.dumps(profile_data, indent=2)
    json_bytes = json_data.encode("utf-8")

    # Upload to MinIO
    try:
        response = s3.put_object(
            Bucket=settings.bucket_name,
            Key=settings.profile_key,
            Body=json_bytes,
            ContentType="application/json",
            Metadata={
                "source": "local_migration",
                "n_clusters": str(profile_data["metadata"]["n_clusters"]),
            },
        )

        version_id = response.get("VersionId", "unversioned")
        size_kb = len(json_bytes) / 1024

        logger.info(
            f"Successfully uploaded to MinIO "
            f"(size: {size_kb:.1f}KB, version: {version_id})"
        )

        return version_id

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(f"MinIO upload failed: {error_code} - {e}")
        raise


def verify_upload(settings: MinIOSettings, original_data: dict) -> bool:
    """Verify uploaded data matches original.

    Args:
        settings: MinIO configuration
        original_data: Original profile data

    Returns:
        True if verification succeeds, False otherwise
    """
    logger.info("Verifying upload...")

    try:
        # Configure S3 client with s3v4 signature for MinIO
        config = Config(
            region_name=settings.region,
            signature_version="s3v4",
        )

        # Download from MinIO
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            config=config,
            aws_access_key_id=settings.root_user,
            aws_secret_access_key=settings.root_password,
        )

        response = s3.get_object(Bucket=settings.bucket_name, Key=settings.profile_key)
        data = response["Body"].read()
        uploaded_data = json.loads(data)

        # Verify key structure
        if uploaded_data.keys() != original_data.keys():
            logger.error("Uploaded data has different keys")
            return False

        # Verify metadata
        if (
            uploaded_data["metadata"]["n_clusters"]
            != original_data["metadata"]["n_clusters"]
        ):
            logger.error("Cluster count mismatch")
            return False

        logger.info("✅ Verification successful - data matches")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main migration function."""
    try:
        # Load MinIO settings from environment
        logger.info("Loading MinIO configuration from environment...")
        try:
            settings = MinIOSettings()
        except Exception as e:
            logger.error(
                f"Failed to load MinIO settings. Make sure environment variables are set:\n"
                f"  - S3_BUCKET_NAME (required)\n"
                f"  - MINIO_PUBLIC_ENDPOINT (required, e.g., https://minio.railway.app)\n"
                f"  - MINIO_ROOT_USER (required)\n"
                f"  - MINIO_ROOT_PASSWORD (required)\n"
                f"Error: {e}"
            )
            sys.exit(1)

        logger.info(
            f"MinIO storage configured: bucket={settings.bucket_name}, endpoint={settings.endpoint_url}"
        )

        # Load local profile data
        # Assuming script is run from project root
        data_dir = Path("adaptive_router/data/unirouter/clusters")
        if not data_dir.exists():
            # Try alternative path (if run from adaptive_router directory)
            data_dir = Path("data/unirouter/clusters")

        if not data_dir.exists():
            logger.error(
                "Cluster data directory not found. "
                "Expected: adaptive_router/data/unirouter/clusters/"
            )
            sys.exit(1)

        profile_data = load_local_profile(data_dir)

        # Upload to MinIO
        version_id = upload_to_minio(profile_data, settings)

        # Verify upload
        if verify_upload(settings, profile_data):
            logger.info(
                f"\n{'='*60}\n"
                f"✅ Migration complete!\n"
                f"{'='*60}\n"
                f"Storage: MinIO\n"
                f"Endpoint: {settings.endpoint_url}\n"
                f"Location: s3://{settings.bucket_name}/{settings.profile_key}\n"
                f"Version ID: {version_id}\n"
                f"Local files: PRESERVED (not deleted)\n"
                f"{'='*60}\n"
            )
            return 0
        else:
            logger.error("Migration completed but verification failed")
            return 1

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
