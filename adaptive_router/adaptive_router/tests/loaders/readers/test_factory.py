"""Tests for readers factory functions."""

import pytest

from adaptive_router.loaders.readers import get_reader, supported_formats
from adaptive_router.loaders.readers.csv import CSVProfileReader
from adaptive_router.loaders.readers.json import JSONProfileReader
from adaptive_router.loaders.readers.parquet import ParquetProfileReader


class TestGetReader:
    """Test get_reader factory function."""

    def test_get_reader_json_extension(self):
        """Test get_reader returns JSONProfileReader for .json extension."""
        reader = get_reader("profile.json")
        assert isinstance(reader, JSONProfileReader)

    def test_get_reader_csv_extension(self):
        """Test get_reader returns CSVProfileReader for .csv extension."""
        reader = get_reader("profile.csv")
        assert isinstance(reader, CSVProfileReader)

    def test_get_reader_parquet_extension(self):
        """Test get_reader returns ParquetProfileReader for .parquet extension."""
        reader = get_reader("profile.parquet")
        assert isinstance(reader, ParquetProfileReader)

    def test_get_reader_pq_extension(self):
        """Test get_reader returns ParquetProfileReader for .pq extension."""
        reader = get_reader("profile.pq")
        assert isinstance(reader, ParquetProfileReader)

    def test_get_reader_case_insensitive(self):
        """Test get_reader is case insensitive."""
        reader = get_reader("profile.JSON")
        assert isinstance(reader, JSONProfileReader)

        reader = get_reader("profile.PARQUET")
        assert isinstance(reader, ParquetProfileReader)

    def test_get_reader_path_object(self):
        """Test get_reader works with Path objects."""
        from pathlib import Path

        reader = get_reader(Path("profile.json"))
        assert isinstance(reader, JSONProfileReader)

    def test_get_reader_unsupported_format(self):
        """Test get_reader raises ValueError for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_reader("profile.txt")

    def test_get_reader_no_extension(self):
        """Test get_reader raises ValueError for files without extension."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_reader("profile")


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
