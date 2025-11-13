"""Tests for writers factory functions."""

from pathlib import Path

import pytest

from adaptive_router.savers.writers import get_writer, supported_formats
from adaptive_router.savers.writers.csv import CSVProfileWriter
from adaptive_router.savers.writers.json import JSONProfileWriter
from adaptive_router.savers.writers.parquet import ParquetProfileWriter


class TestGetWriter:
    """Test get_writer factory function."""

    def test_get_writer_json_extension(self):
        """Test get_writer returns JSONProfileWriter for .json extension."""
        writer = get_writer("profile.json")
        assert isinstance(writer, JSONProfileWriter)

    def test_get_writer_csv_extension(self):
        """Test get_writer returns CSVProfileWriter for .csv extension."""
        writer = get_writer("profile.csv")
        assert isinstance(writer, CSVProfileWriter)

    def test_get_writer_parquet_extension(self):
        """Test get_writer returns ParquetProfileWriter for .parquet extension."""
        writer = get_writer("profile.parquet")
        assert isinstance(writer, ParquetProfileWriter)

    def test_get_writer_pq_extension(self):
        """Test get_writer returns ParquetProfileWriter for .pq extension."""
        writer = get_writer("profile.pq")
        assert isinstance(writer, ParquetProfileWriter)

    def test_get_writer_case_insensitive(self):
        """Test get_writer is case insensitive."""
        writer = get_writer("profile.JSON")
        assert isinstance(writer, JSONProfileWriter)

        writer = get_writer("profile.CSV")
        assert isinstance(writer, CSVProfileWriter)

        writer = get_writer("profile.PARQUET")
        assert isinstance(writer, ParquetProfileWriter)

    def test_get_writer_path_object(self):
        """Test get_writer works with Path objects."""
        writer = get_writer(Path("profile.json"))
        assert isinstance(writer, JSONProfileWriter)

    def test_get_writer_unsupported_format(self):
        """Test get_writer raises ValueError for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_writer("profile.txt")


class TestSupportedFormats:
    """Test supported_formats function."""

    def test_supported_formats_includes_json(self):
        """Test supported_formats includes .json."""
        formats = supported_formats()
        assert ".json" in formats

    def test_supported_formats_includes_csv(self):
        """Test supported_formats includes .csv."""
        formats = supported_formats()
        assert ".csv" in formats

    def test_supported_formats_includes_parquet(self):
        """Test supported_formats includes .parquet."""
        formats = supported_formats()
        assert ".parquet" in formats

    def test_supported_formats_includes_pq(self):
        """Test supported_formats includes .pq."""
        formats = supported_formats()
        assert ".pq" in formats

    def test_supported_formats_returns_list(self):
        """Test supported_formats returns a list."""
        formats = supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0