"""Tests for registry models and exceptions."""

import pytest
from pydantic import ValidationError

from app.models import (
    RegistryModel,
    RegistryError,
    RegistryClientConfig,
    RegistryConnectionError,
    RegistryResponseError,
)


class TestRegistryModel:
    """Test RegistryModel."""

    def test_minimal_registry_model(self) -> None:
        """Test creating RegistryModel with minimal fields."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.pricing is None
        assert model.context_length is None

    def test_full_registry_model(self) -> None:
        """Test creating RegistryModel with all fields."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            description="GPT-4 model",
            context_length=128000,
            pricing={"prompt_cost": "0.00003", "completion_cost": "0.00006"},
            supported_parameters=[
                {"parameter_name": "temperature"},
                {"parameter_name": "max_tokens"},
            ],
        )
        assert model.provider == "openai"
        assert model.description == "GPT-4 model"
        assert model.context_length == 128000
        assert model.pricing is not None

    def test_unique_id_generation(self) -> None:
        """Test unique_id generates correct format."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.unique_id() == "openai:gpt-4"

    def test_unique_id_lowercase_normalization(self) -> None:
        """Test unique_id normalizes to lowercase."""
        model = RegistryModel(provider="OpenAI", model_name="GPT-4")
        assert model.unique_id() == "openai:gpt-4"

    def test_unique_id_missing_provider(self) -> None:
        """Test unique_id raises error when provider is empty."""
        model = RegistryModel(provider="", model_name="gpt-4")
        with pytest.raises(RegistryError, match="missing provider"):
            model.unique_id()

    def test_unique_id_missing_model_name(self) -> None:
        """Test unique_id raises error when model_name is empty."""
        model = RegistryModel(provider="openai", model_name="")
        with pytest.raises(RegistryError, match="missing model_name"):
            model.unique_id()

    def test_average_price_calculation(self) -> None:
        """Test average_price calculates correctly."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "0.00003", "completion_cost": "0.00006"},
        )
        avg = model.average_price()
        assert avg is not None
        assert avg == pytest.approx(0.000045)

    def test_average_price_no_pricing(self) -> None:
        """Test average_price returns None when no pricing."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.average_price() is None

    def test_average_price_zero_costs(self) -> None:
        """Test average_price returns None for zero costs."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "0", "completion_cost": "0"},
        )
        assert model.average_price() is None

    def test_average_price_invalid_pricing(self) -> None:
        """Test average_price handles invalid pricing strings."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "invalid", "completion_cost": "0.00006"},
        )
        assert model.average_price() is None


class TestRegistryClientConfig:
    """Test RegistryClientConfig."""

    def test_default_config(self) -> None:
        """Test RegistryClientConfig with default values."""
        config = RegistryClientConfig(base_url="http://localhost:8000")
        assert config.base_url == "http://localhost:8000"
        assert config.timeout == 5.0
        assert config.default_headers is None

    def test_custom_config(self) -> None:
        """Test RegistryClientConfig with custom values."""
        config = RegistryClientConfig(
            base_url="http://api.example.com",
            timeout=60.0,
            default_headers={"Authorization": "Bearer token"},
        )
        assert config.base_url == "http://api.example.com"
        assert config.timeout == 60.0
        assert config.default_headers == {"Authorization": "Bearer token"}

    def test_config_validation(self) -> None:
        """Test RegistryClientConfig validation."""
        # base_url is required
        with pytest.raises(ValidationError):
            RegistryClientConfig()  # type: ignore

    def test_normalized_headers(self) -> None:
        """Test normalized_headers method."""
        config = RegistryClientConfig(
            base_url="http://localhost:8000",
            default_headers={"X-Custom": "value"},
        )
        headers = config.normalized_headers()
        assert headers == {"X-Custom": "value"}

    def test_normalized_headers_none(self) -> None:
        """Test normalized_headers with no headers."""
        config = RegistryClientConfig(base_url="http://localhost:8000")
        headers = config.normalized_headers()
        assert headers == {}


class TestRegistryExceptions:
    """Test registry exception classes."""

    def test_registry_error(self) -> None:
        """Test RegistryError exception."""
        error = RegistryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_registry_connection_error(self) -> None:
        """Test RegistryConnectionError exception."""
        error = RegistryConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, RegistryError)

    def test_registry_response_error(self) -> None:
        """Test RegistryResponseError exception."""
        error = RegistryResponseError("Invalid response")
        assert str(error) == "Invalid response"
        assert isinstance(error, RegistryError)

    def test_exception_hierarchy(self) -> None:
        """Test exception hierarchy."""
        # RegistryConnectionError and RegistryResponseError are both RegistryErrors
        conn_error = RegistryConnectionError("test")
        resp_error = RegistryResponseError("test")

        assert isinstance(conn_error, RegistryError)
        assert isinstance(resp_error, RegistryError)
        assert isinstance(conn_error, Exception)
        assert isinstance(resp_error, Exception)
