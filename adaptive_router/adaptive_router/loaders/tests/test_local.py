"""Tests for LocalFileProfileLoader."""

import json
from pathlib import Path

import pytest

from adaptive_router.loaders.local import LocalFileProfileLoader


@pytest.fixture
def valid_profile_data() -> dict:
    """Create valid profile data for testing."""
    # Full RouterProfile structure with all required fields
    return {
        "metadata": {
            "n_clusters": 2,
            "n_samples": 10,
            "silhouette_score": 0.45,
            "created_at": "2024-01-01T00:00:00Z",
            "model_version": "1.0.0",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "tfidf_max_features": 5000,
        },
        "cluster_centers": {
            "centers": [[0.1, 0.2], [0.4, 0.5]],
            "n_clusters": 2,
            "feature_dimension": 2,
        },
        "llm_profiles": {
            "openai:gpt-4": [0.05, 0.03],
            "anthropic:claude-3-sonnet": [0.07, 0.04],
        },
        "tfidf_vocabulary": {
            "vocabulary": {"python": 0, "javascript": 1},
            "idf_scores": [1.0, 1.2],
        },
        "scaler_parameters": {
            "mean": [0.5, 0.5],
            "scale": [0.2, 0.2],
            "n_features": 2,
        },
    }


@pytest.fixture
def profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary profile file."""
    file_path = tmp_path / "test_profile.json"
    with open(file_path, "w") as f:
        json.dump(valid_profile_data, f)
    return file_path


class TestLocalFileProfileLoaderInitialization:
    """Test LocalFileProfileLoader initialization."""

    def test_initialization_with_valid_path(self, profile_file: Path) -> None:
        """Test loader initializes with valid file path."""
        loader = LocalFileProfileLoader(profile_file)
        assert loader.profile_path == profile_file

    def test_initialization_with_string_path(self, profile_file: Path) -> None:
        """Test loader accepts string path."""
        loader = LocalFileProfileLoader(str(profile_file))
        assert loader.profile_path == profile_file

    def test_initialization_with_nonexistent_path(self, tmp_path) -> None:
        """Test loader raises error for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Profile file not found"):
            LocalFileProfileLoader(nonexistent)


class TestLocalFileProfileLoaderLoadProfile:
    """Test LocalFileProfileLoader load_profile method."""

    # Note: Full RouterProfile schema is complex (requires cluster_centers, llm_profiles,
    # tfidf_vocabulary, scaler_parameters with specific nested structures).
    # We test error paths which are more critical for robustness.

    def test_load_profile_with_corrupted_json(self, tmp_path) -> None:
        """Test loading corrupted JSON raises ValueError."""
        corrupted_file = tmp_path / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write("{invalid json content")

        loader = LocalFileProfileLoader(corrupted_file)
        with pytest.raises(ValueError, match="Corrupted JSON"):
            loader.load_profile()

    def test_load_profile_with_invalid_schema(self, tmp_path) -> None:
        """Test loading profile with invalid schema raises ValueError."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump({"invalid": "schema"}, f)

        loader = LocalFileProfileLoader(invalid_file)
        with pytest.raises(ValueError, match="validation"):
            loader.load_profile()


class TestLocalFileProfileLoaderHealthCheck:
    """Test LocalFileProfileLoader health_check method."""

    def test_health_check_with_valid_file(self, profile_file: Path) -> None:
        """Test health check returns True for valid file."""
        loader = LocalFileProfileLoader(profile_file)
        assert loader.health_check() is True

    def test_health_check_after_file_deletion(self, profile_file: Path) -> None:
        """Test health check returns False after file deletion."""
        loader = LocalFileProfileLoader(profile_file)
        assert loader.health_check() is True

        # Delete the file
        profile_file.unlink()
        assert loader.health_check() is False

    def test_health_check_with_directory_path(self, tmp_path) -> None:
        """Test health check returns False for directory."""
        # Create a directory with the profile name
        dir_path = tmp_path / "profile.json"
        dir_path.mkdir()

        # Create a dummy file to allow initialization
        dummy_file = tmp_path / "dummy.json"
        with open(dummy_file, "w") as f:
            json.dump({"test": "data"}, f)

        loader = LocalFileProfileLoader(dummy_file)
        loader.profile_path = dir_path  # Override with directory

        assert loader.health_check() is False
