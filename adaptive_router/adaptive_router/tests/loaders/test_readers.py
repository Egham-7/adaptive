"""Tests for profile readers."""

import json

import pandas as pd
import pytest

from adaptive_router.loaders.readers import (
    CSVProfileReader,
    JSONProfileReader,
    ParquetProfileReader,
    get_reader,
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


class TestJSONProfileReader:
    """Test JSONProfileReader."""

    def test_read_from_path_valid_json(self, tmp_path, valid_profile_data):
        """Test reading valid JSON file."""
        json_file = tmp_path / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(valid_profile_data, f, indent=2)

        reader = JSONProfileReader()
        profile = reader.read_from_path(json_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2
        assert len(profile.models) == 1

    def test_read_from_bytes_valid_json(self, valid_profile_data):
        """Test reading valid JSON bytes."""
        json_bytes = json.dumps(valid_profile_data, indent=2).encode("utf-8")

        reader = JSONProfileReader()
        profile = reader.read_from_bytes(json_bytes)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2

    def test_read_from_path_invalid_json(self, tmp_path):
        """Test reading invalid JSON raises ValueError."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w") as f:
            f.write("{invalid json")

        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.read_from_path(json_file)

    def test_read_from_bytes_invalid_json(self):
        """Test reading invalid JSON bytes raises ValueError."""
        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.read_from_bytes(b"{invalid json")

    def test_read_from_path_invalid_profile_schema(self, tmp_path):
        """Test reading JSON with invalid profile schema raises ValueError."""
        json_file = tmp_path / "invalid.json"
        with open(json_file, "w") as f:
            json.dump({"invalid": "schema"}, f)

        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="validation"):
            reader.read_from_path(json_file)

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert JSONProfileReader.supported_extensions() == [".json"]


class TestCSVProfileReader:
    """Test CSVProfileReader."""

    def test_read_from_path_valid_csv(self, tmp_path, valid_profile_data):
        """Test reading valid CSV file."""
        csv_file = tmp_path / "test.csv"

        # Create CSV with format and data columns
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=["format", "data"])
            writer.writeheader()
            writer.writerow({"format": "json", "data": json.dumps(valid_profile_data)})

        reader = CSVProfileReader()
        profile = reader.read_from_path(csv_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2

    def test_read_from_bytes_valid_csv(self, valid_profile_data):
        """Test reading valid CSV bytes."""
        # Create CSV content
        import io

        output = io.StringIO()
        import csv

        writer = csv.DictWriter(output, fieldnames=["format", "data"])
        writer.writeheader()
        writer.writerow({"format": "json", "data": json.dumps(valid_profile_data)})
        csv_content = output.getvalue()

        reader = CSVProfileReader()
        profile = reader.read_from_bytes(csv_content.encode("utf-8"))

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2

    def test_read_from_path_missing_format_column(self, tmp_path, valid_profile_data):
        """Test reading CSV without format column raises ValueError."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=["data"])
            writer.writeheader()
            writer.writerow({"data": json.dumps(valid_profile_data)})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_path(csv_file)

    def test_read_from_path_wrong_format(self, tmp_path, valid_profile_data):
        """Test reading CSV with wrong format raises ValueError."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=["format", "data"])
            writer.writeheader()
            writer.writerow({"format": "xml", "data": json.dumps(valid_profile_data)})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Unsupported format 'xml'"):
            reader.read_from_path(csv_file)

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert CSVProfileReader.supported_extensions() == [".csv"]


class TestParquetProfileReader:
    """Test ParquetProfileReader."""

    def test_read_from_path_valid_parquet(self, tmp_path, valid_profile_data):
        """Test reading valid Parquet file."""
        parquet_file = tmp_path / "test.parquet"

        # Create DataFrame and save as Parquet
        df = pd.DataFrame([{"format": "json", "data": json.dumps(valid_profile_data)}])
        df.to_parquet(parquet_file, index=False)

        reader = ParquetProfileReader()
        profile = reader.read_from_path(parquet_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2

    def test_read_from_bytes_valid_parquet(self, valid_profile_data):
        """Test reading valid Parquet bytes."""
        # Create DataFrame
        df = pd.DataFrame([{"format": "json", "data": json.dumps(valid_profile_data)}])

        # Convert to bytes
        import io

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        profile = reader.read_from_bytes(parquet_bytes)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == 2

    def test_read_from_path_missing_format_column(self, tmp_path, valid_profile_data):
        """Test reading Parquet without format column raises ValueError."""
        parquet_file = tmp_path / "test.parquet"

        df = pd.DataFrame([{"data": json.dumps(valid_profile_data)}])
        df.to_parquet(parquet_file, index=False)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_path(parquet_file)

    def test_supported_extensions(self):
        """Test supported extensions."""
        assert ParquetProfileReader.supported_extensions() == [".parquet", ".pq"]


class TestReaderRegistry:
    """Test reader registry and factory functions."""

    def test_get_reader_json(self, tmp_path):
        """Test get_reader returns JSONProfileReader for .json files."""
        json_file = tmp_path / "test.json"
        json_file.touch()

        reader = get_reader(json_file)
        assert isinstance(reader, JSONProfileReader)

    def test_get_reader_csv(self, tmp_path):
        """Test get_reader returns CSVProfileReader for .csv files."""
        csv_file = tmp_path / "test.csv"
        csv_file.touch()

        reader = get_reader(csv_file)
        assert isinstance(reader, CSVProfileReader)

    def test_get_reader_parquet(self, tmp_path):
        """Test get_reader returns ParquetProfileReader for .parquet files."""
        parquet_file = tmp_path / "test.parquet"
        parquet_file.touch()

        reader = get_reader(parquet_file)
        assert isinstance(reader, ParquetProfileReader)

    def test_get_reader_pq_extension(self, tmp_path):
        """Test get_reader returns ParquetProfileReader for .pq files."""
        pq_file = tmp_path / "test.pq"
        pq_file.touch()

        reader = get_reader(pq_file)
        assert isinstance(reader, ParquetProfileReader)

    def test_get_reader_unsupported_format(self, tmp_path):
        """Test get_reader raises ValueError for unsupported formats."""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match="Unsupported format"):
            get_reader(txt_file)

    def test_supported_formats(self):
        """Test supported_formats returns all supported extensions."""
        formats = supported_formats()
        expected = [".json", ".csv", ".parquet", ".pq"]
        assert set(formats) == set(expected)
