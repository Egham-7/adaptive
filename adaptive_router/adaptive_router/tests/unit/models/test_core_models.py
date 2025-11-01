"""Tests for core API models."""

from pydantic import ValidationError
import pytest

from adaptive_router.models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.registry import RegistryModel, RegistryError


class TestRegistryModel:
    """Test RegistryModel validation and functionality."""

    def test_minimal_registry_model(self) -> None:
        """Test creating RegistryModel with minimal required fields."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.pricing is None

    def test_full_registry_model(self) -> None:
        """Test creating RegistryModel with all fields."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            description="GPT-4 model",
            context_length=128000,
            pricing={"prompt": "0.00003", "completion": "0.00006"},
        )
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.description == "GPT-4 model"
        assert model.context_length == 128000
        assert model.pricing == {"prompt": "0.00003", "completion": "0.00006"}

    def test_unique_id(self) -> None:
        """Test unique_id generation."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.unique_id() == "openai:gpt-4"

    def test_unique_id_with_provider_prefix(self) -> None:
        """Test unique_id removes provider prefix."""
        model = RegistryModel(provider="openai", model_name="openai/gpt-4")
        assert model.unique_id() == "openai:gpt-4"

    def test_unique_id_missing_provider(self) -> None:
        """Test unique_id raises error when provider is missing."""
        model = RegistryModel(provider="", model_name="gpt-4")
        with pytest.raises(RegistryError, match="missing provider"):
            model.unique_id()

    def test_average_price(self) -> None:
        """Test average_price calculation."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt": "0.00003", "completion": "0.00006"},
        )
        avg = model.average_price()
        assert avg is not None
        assert avg == pytest.approx(0.000045)

    def test_average_price_no_pricing(self) -> None:
        """Test average_price returns None when no pricing."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.average_price() is None

    def test_average_price_zero_costs(self) -> None:
        """Test average_price returns None when costs are zero."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt": "0", "completion": "0"},
        )
        assert model.average_price() is None


class TestModelSelectionRequest:
    """Test ModelSelectionRequest validation."""

    def test_valid_model_selection_request(self) -> None:
        """Test creating valid ModelSelectionRequest."""
        request = ModelSelectionRequest(
            prompt="Test prompt",
            cost_bias=0.5,
        )
        assert request.prompt == "Test prompt"
        assert request.cost_bias == 0.5

    def test_prompt_validation_empty(self) -> None:
        """Test prompt cannot be empty."""
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="")

    def test_prompt_validation_whitespace(self) -> None:
        """Test prompt cannot be only whitespace."""
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="   ")

    def test_cost_bias_validation(self) -> None:
        """Test cost_bias must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=1.5)

        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=-0.1)

    def test_with_registry_models(self) -> None:
        """Test ModelSelectionRequest with RegistryModel list."""
        models = [
            RegistryModel(provider="openai", model_name="gpt-4"),
            RegistryModel(provider="anthropic", model_name="claude-3-sonnet"),
        ]
        request = ModelSelectionRequest(prompt="Test prompt", models=models)
        assert len(request.models) == 2
        assert request.models[0].provider == "openai"


class TestModelSelectionResponse:
    """Test ModelSelectionResponse validation."""

    def test_valid_response(self) -> None:
        """Test creating valid ModelSelectionResponse."""
        response = ModelSelectionResponse(
            provider="openai",
            model="gpt-4",
            alternatives=[
                Alternative(provider="anthropic", model="claude-3-sonnet"),
            ],
        )
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert len(response.alternatives) == 1

    def test_empty_provider_validation(self) -> None:
        """Test provider cannot be empty."""
        with pytest.raises(ValidationError):
            ModelSelectionResponse(
                provider="",
                model="gpt-4",
                alternatives=[],
            )

    def test_empty_model_validation(self) -> None:
        """Test model cannot be empty."""
        with pytest.raises(ValidationError):
            ModelSelectionResponse(
                provider="openai",
                model="",
                alternatives=[],
            )


class TestAlternative:
    """Test Alternative model."""

    def test_valid_alternative(self) -> None:
        """Test creating valid Alternative."""
        alt = Alternative(provider="openai", model="gpt-3.5-turbo")
        assert alt.provider == "openai"
        assert alt.model == "gpt-3.5-turbo"

    def test_empty_provider_validation(self) -> None:
        """Test provider cannot be empty."""
        with pytest.raises(ValidationError):
            Alternative(provider="", model="gpt-4")

    def test_empty_model_validation(self) -> None:
        """Test model cannot be empty."""
        with pytest.raises(ValidationError):
            Alternative(provider="openai", model="")
