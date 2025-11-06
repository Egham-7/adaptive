"""Tests for model_parser utility functions."""

import pytest

from adaptive_router.utils.model_parser import parse_model_spec


class TestParseModelSpec:
    """Test parse_model_spec function."""

    def test_parse_valid_model_spec(self) -> None:
        """Test parsing valid model specification."""
        provider, model_name = parse_model_spec("openai:gpt-4")
        assert provider == "openai"
        assert model_name == "gpt-4"

    def test_parse_with_whitespace(self) -> None:
        """Test parsing trims whitespace."""
        provider, model_name = parse_model_spec("  openai : gpt-4  ")
        assert provider == "openai"
        assert model_name == "gpt-4"

    def test_parse_with_hyphens(self) -> None:
        """Test parsing model names with hyphens."""
        provider, model_name = parse_model_spec("anthropic:claude-3-sonnet-20240229")
        assert provider == "anthropic"
        assert model_name == "claude-3-sonnet-20240229"

    def test_parse_with_dots(self) -> None:
        """Test parsing model names with dots."""
        provider, model_name = parse_model_spec("google:gemini-1.5-pro")
        assert provider == "google"
        assert model_name == "gemini-1.5-pro"

    def test_parse_empty_string_raises_error(self) -> None:
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_model_spec("")

    def test_parse_whitespace_only_raises_error(self) -> None:
        """Test parsing whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_model_spec("   ")

    def test_parse_no_colon_raises_error(self) -> None:
        """Test parsing string without colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model specification format"):
            parse_model_spec("openai-gpt-4")

    def test_parse_multiple_colons_raises_error(self) -> None:
        """Test parsing string with multiple colons raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model specification format"):
            parse_model_spec("openai:gpt:4")

    def test_parse_empty_provider_raises_error(self) -> None:
        """Test parsing with empty provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            parse_model_spec(":gpt-4")

    def test_parse_whitespace_provider_raises_error(self) -> None:
        """Test parsing with whitespace-only provider raises ValueError."""
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            parse_model_spec("  :gpt-4")

    def test_parse_empty_model_name_raises_error(self) -> None:
        """Test parsing with empty model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            parse_model_spec("openai:")

    def test_parse_whitespace_model_name_raises_error(self) -> None:
        """Test parsing with whitespace-only model name raises ValueError."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            parse_model_spec("openai:  ")


class TestParseModelSpecEdgeCases:
    """Test edge cases for parse_model_spec."""

    def test_parse_with_numbers(self) -> None:
        """Test parsing with numeric characters."""
        provider, model_name = parse_model_spec("openai:gpt-4-turbo-2024-04-09")
        assert provider == "openai"
        assert model_name == "gpt-4-turbo-2024-04-09"

    def test_parse_with_underscores(self) -> None:
        """Test parsing model names with underscores."""
        provider, model_name = parse_model_spec("mistral:mistral_7b_instruct")
        assert provider == "mistral"
        assert model_name == "mistral_7b_instruct"

    def test_parse_short_names(self) -> None:
        """Test parsing very short provider and model names."""
        provider, model_name = parse_model_spec("a:b")
        assert provider == "a"
        assert model_name == "b"

    def test_parse_long_names(self) -> None:
        """Test parsing very long model specifications."""
        long_spec = "very-long-provider-name:very-long-model-name-with-many-segments"
        provider, model_name = parse_model_spec(long_spec)
        assert provider == "very-long-provider-name"
        assert model_name == "very-long-model-name-with-many-segments"
