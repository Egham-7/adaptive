"""Tests for RegistryClient input validation and whitespace handling."""

from unittest.mock import Mock, patch

import httpx
import pytest
from requests import Response

from adaptive_router.models.registry import RegistryClientConfig
from app.registry.client import RegistryClient


@pytest.fixture
def mock_response() -> Mock:
    """Mock successful HTTP response."""
    response = Mock(spec=Response)
    response.status_code = 200
    response.json.return_value = {
        "provider": "openai",
        "model_name": "gpt-4",
        "pricing": {"prompt_cost": "0.00003", "completion_cost": "0.00006"},
        "context_length": 128000,
    }
    return response


@pytest.fixture
def client() -> RegistryClient:
    """Create a RegistryClient instance."""
    config = RegistryClientConfig(base_url="http://test-registry.example.com")
    http_client = httpx.Client(timeout=5.0)
    return RegistryClient(config=config, client=http_client)


class TestWhitespaceHandling:
    """Test whitespace trimming in get_by_provider_and_name."""

    @patch.object(RegistryClient, "_request")
    def test_trims_provider_whitespace(
        self, mock_request: Mock, client: RegistryClient, mock_response: Mock
    ) -> None:
        """Test that leading/trailing whitespace is stripped from provider."""
        mock_request.return_value = mock_response

        # Call with whitespace-padded provider
        result = client.get_by_provider_and_name("  openai  ", "gpt-4")

        assert result is not None
        assert result.provider == "openai"
        # Verify the URL was constructed with trimmed values
        mock_request.assert_called_once()
        assert "/models/openai/gpt-4" in mock_request.call_args[0][1]

    @patch.object(RegistryClient, "_request")
    def test_trims_name_whitespace(
        self, mock_request: Mock, client: RegistryClient, mock_response: Mock
    ) -> None:
        """Test that leading/trailing whitespace is stripped from name."""
        mock_request.return_value = mock_response

        # Call with whitespace-padded name
        result = client.get_by_provider_and_name("openai", "  gpt-4  ")

        assert result is not None
        assert result.model_name == "gpt-4"
        # Verify the URL was constructed with trimmed values
        mock_request.assert_called_once()
        assert "/models/openai/gpt-4" in mock_request.call_args[0][1]

    @patch.object(RegistryClient, "_request")
    def test_trims_both_parameters(
        self, mock_request: Mock, client: RegistryClient, mock_response: Mock
    ) -> None:
        """Test that whitespace is stripped from both parameters."""
        mock_request.return_value = mock_response

        # Call with whitespace on both parameters
        result = client.get_by_provider_and_name("  openai  ", "  gpt-4  ")

        assert result is not None
        # Verify the URL was constructed with trimmed values
        mock_request.assert_called_once()
        assert "/models/openai/gpt-4" in mock_request.call_args[0][1]


class TestEmptyInputValidation:
    """Test validation of empty inputs after trimming."""

    def test_raises_on_empty_provider(self, client: RegistryClient) -> None:
        """Test that empty provider raises ValueError."""
        with pytest.raises(ValueError, match="provider must be provided"):
            client.get_by_provider_and_name("", "gpt-4")

    def test_raises_on_whitespace_only_provider(self, client: RegistryClient) -> None:
        """Test that whitespace-only provider raises ValueError after trimming."""
        with pytest.raises(ValueError, match="provider must be provided"):
            client.get_by_provider_and_name("   ", "gpt-4")

    def test_raises_on_empty_name(self, client: RegistryClient) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name must be provided"):
            client.get_by_provider_and_name("openai", "")

    def test_raises_on_whitespace_only_name(self, client: RegistryClient) -> None:
        """Test that whitespace-only name raises ValueError after trimming."""
        with pytest.raises(ValueError, match="name must be provided"):
            client.get_by_provider_and_name("openai", "   ")

    def test_raises_on_none_provider(self, client: RegistryClient) -> None:
        """Test that None provider raises ValueError."""
        with pytest.raises(ValueError, match="provider must be provided"):
            client.get_by_provider_and_name(None, "gpt-4")  # type: ignore

    def test_raises_on_none_name(self, client: RegistryClient) -> None:
        """Test that None name raises ValueError."""
        with pytest.raises(ValueError, match="name must be provided"):
            client.get_by_provider_and_name("openai", None)  # type: ignore
