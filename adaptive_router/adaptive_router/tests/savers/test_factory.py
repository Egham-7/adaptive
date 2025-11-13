"""Tests for savers factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from adaptive_router.savers import get_saver


class TestGetSaver:
    """Test get_saver factory function."""

    def test_get_saver_local_path(self):
        """Test get_saver returns LocalFileProfileSaver for local paths."""
        from adaptive_router.savers.local import LocalFileProfileSaver

        saver = get_saver("profile.json")
        assert isinstance(saver, LocalFileProfileSaver)

    def test_get_saver_s3_path_without_settings(self):
        """Test get_saver raises error for S3 paths without settings."""
        with pytest.raises(ValueError, match="S3 settings required"):
            get_saver("s3://bucket/profile.json")

    def test_get_saver_minio_path_without_settings(self):
        """Test get_saver raises error for MinIO paths without settings."""
        with pytest.raises(ValueError, match="MinIO settings required"):
            get_saver("minio://bucket/profile.json")

    @patch('adaptive_router.savers.MinIOProfileSaver')
    def test_get_saver_s3_path_with_settings(self, mock_minio_saver):
        """Test get_saver returns MinIOProfileSaver for S3 paths with settings."""
        from adaptive_router.models.storage import MinIOSettings

        settings = MinIOSettings(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profile.json",
            endpoint_url="https://s3.amazonaws.com",
            root_user="test",
            root_password="test"
        )

        mock_instance = MagicMock()
        mock_minio_saver.from_settings.return_value = mock_instance

        result = get_saver("s3://bucket/profile.json", s3_settings=settings)

        mock_minio_saver.from_settings.assert_called_once_with(settings)
        assert result == mock_instance

    @patch('adaptive_router.savers.MinIOProfileSaver')
    def test_get_saver_minio_path_with_settings(self, mock_minio_saver):
        """Test get_saver returns MinIOProfileSaver for MinIO paths with settings."""
        from adaptive_router.models.storage import MinIOSettings

        settings = MinIOSettings(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profile.json",
            endpoint_url="https://minio.example.com",
            root_user="test",
            root_password="test"
        )

        mock_instance = MagicMock()
        mock_minio_saver.from_settings.return_value = mock_instance

        result = get_saver("minio://bucket/profile.json", minio_settings=settings)

        mock_minio_saver.from_settings.assert_called_once_with(settings)
        assert result == mock_instance