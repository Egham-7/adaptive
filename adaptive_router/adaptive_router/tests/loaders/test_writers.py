"""Tests for profile writers."""

import json

import pandas as pd
import pytest

from adaptive_router.loaders.writers import (
    CSVProfileWriter,
    JSONProfileWriter,
    ParquetProfileWriter,
    get_writer,
    supported_formats,
)
from adaptive_router.models.storage import RouterProfile


@pytest.fixture
def valid_profile_data() -> dict:
    """Create valid profile data for testing."""
    return {
        "metadata": {
            "n_clusters": 2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "tfidf_max_features": 5000,
            "tfidf_ngram_range": [1, 2],
            "silhouette_score": 0.45,
        },
        "cluster_centers": {
            "n_clusters": 2,
            "feature_dim": 2,
            "cluster_centers": [[0.1, 0.2], [0.4, 0.5]],
        },
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
            }
        ],
        "llm_profiles": {
            "openai/gpt-4": [0.05, 0.03],
        },
        "tfidf_vocabulary": {
            "vocabulary": {"python": 0, "javascript": 1},
            "idf": [1.0, 1.2],
        },
        "scaler_parameters": {
            "embedding_scaler": {
                "mean": [0.5, 0.5],
                "scale": [0.2, 0.2],
            },
            "tfidf_scaler": {
                "mean": [0.3, 0.3],
                "scale": [0.1, 0.1],
            },
        },
    }


@pytest.fixture
def valid_profile(valid_profile_data) -> RouterProfile:
    """Create a valid RouterProfile object."""
    return RouterProfile(**valid_profile_data)


class TestJSONProfileWriter:
    """Test JSONProfileWriter."""

    def test_write_to_path_creates_file(self, tmp_path, valid_profile):
        """Test writing to path creates JSON file."""
        json_file = tmp_path / "test.json"

        writer = JSONProfileWriter()
        writer.write_to_path(valid_profile, json_file)

        assert json_file.exists()
        assert json_file.stat().st_size > 0

        # Verify content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["n_clusters"] == 2
        assert len(data["models"]) == 1

    def test_write_to_bytes_returns_bytes(self, valid_profile):
        """Test writing to bytes returns valid JSON bytes."""
        writer = JSONProfileWriter()
        data = writer.write_to_bytes(valid_profile)

        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify content
        json_str = data.decode("utf-8")
        profile_data = json.loads(json_str)
        assert profile_data["metadata"]["n_clusters"] == 2

    def test_write_to_path_creates_parent_dirs(self, tmp_path, valid_profile):
        """Test writing creates parent directories."""
        nested_file = tmp_path / "subdir" / "nested" / "test.json"

        writer = JSONProfileWriter()
        writer.write_to_path(valid_profile, nested_file)

        assert nested_file.exists()
        assert nested_file.parent.exists()

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert JSONProfileWriter.supported_extensions() == [".json"]


class TestCSVProfileWriter:
    """Test CSVProfileWriter."""

    def test_write_to_path_creates_file(self, tmp_path, valid_profile):
        """Test writing to path creates CSV file."""
        csv_file = tmp_path / "test.csv"

        writer = CSVProfileWriter()
        writer.write_to_path(valid_profile, csv_file)

        assert csv_file.exists()
        assert csv_file.stat().st_size > 0

        # Verify content
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            import csv

            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["format"] == "json"

        # Verify JSON data
        profile_data = json.loads(row["data"])
        assert profile_data["metadata"]["n_clusters"] == 2

    def test_write_to_bytes_returns_bytes(self, valid_profile):
        """Test writing to bytes returns valid CSV bytes."""
        writer = CSVProfileWriter()
        data = writer.write_to_bytes(valid_profile)

        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify content
        csv_content = data.decode("utf-8")
        import csv

        reader = csv.DictReader(csv_content.splitlines())
        rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["format"] == "json"

        profile_data = json.loads(row["data"])
        assert profile_data["metadata"]["n_clusters"] == 2

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert CSVProfileWriter.supported_extensions() == [".csv"]


class TestParquetProfileWriter:
    """Test ParquetProfileWriter."""

    def test_write_to_path_creates_file(self, tmp_path, valid_profile):
        """Test writing to path creates Parquet file."""
        parquet_file = tmp_path / "test.parquet"

        writer = ParquetProfileWriter()
        writer.write_to_path(valid_profile, parquet_file)

        assert parquet_file.exists()
        assert parquet_file.stat().st_size > 0

        # Verify content
        df = pd.read_parquet(parquet_file)
        assert len(df) == 1

        row = df.iloc[0]
        assert row["format"] == "json"

        # Verify JSON data
        profile_data = json.loads(row["data"])
        assert profile_data["metadata"]["n_clusters"] == 2

    def test_write_to_bytes_returns_bytes(self, valid_profile):
        """Test writing to bytes returns valid Parquet bytes."""
        writer = ParquetProfileWriter()
        data = writer.write_to_bytes(valid_profile)

        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify content
        import io

        df = pd.read_parquet(io.BytesIO(data))
        assert len(df) == 1

        row = df.iloc[0]
        assert row["format"] == "json"

        profile_data = json.loads(row["data"])
        assert profile_data["metadata"]["n_clusters"] == 2

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert ParquetProfileWriter.supported_extensions() == [".parquet", ".pq"]


class TestWriterRegistry:
    """Test writer registry and factory functions."""

    def test_get_writer_json(self, tmp_path):
        """Test get_writer returns JSONProfileWriter for .json files."""
        json_file = tmp_path / "test.json"

        writer = get_writer(json_file)
        assert isinstance(writer, JSONProfileWriter)

    def test_get_writer_csv(self, tmp_path):
        """Test get_writer returns CSVProfileWriter for .csv files."""
        csv_file = tmp_path / "test.csv"

        writer = get_writer(csv_file)
        assert isinstance(writer, CSVProfileWriter)

    def test_get_writer_parquet(self, tmp_path):
        """Test get_writer returns ParquetProfileWriter for .parquet files."""
        parquet_file = tmp_path / "test.parquet"

        writer = get_writer(parquet_file)
        assert isinstance(writer, ParquetProfileWriter)

    def test_get_writer_pq_extension(self, tmp_path):
        """Test get_writer returns ParquetProfileWriter for .pq files."""
        pq_file = tmp_path / "test.pq"

        writer = get_writer(pq_file)
        assert isinstance(writer, ParquetProfileWriter)

    def test_get_writer_unsupported_format(self, tmp_path):
        """Test get_writer raises ValueError for unsupported formats."""
        txt_file = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            get_writer(txt_file)

    def test_supported_formats(self):
        """Test supported_formats returns all supported extensions."""
        formats = supported_formats()
        expected = [".json", ".csv", ".parquet", ".pq"]
        assert set(formats) == set(expected)
