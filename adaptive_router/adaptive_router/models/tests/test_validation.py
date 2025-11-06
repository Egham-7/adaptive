"""Enhanced model validation tests for comprehensive edge case coverage."""

from pydantic import ValidationError
import pytest

from adaptive_router.models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.registry import RegistryModel, RegistryError


class TestRegistryModelValidation:
    """Test RegistryModel validation and edge cases."""

    def test_partial_registry_model_validation(self) -> None:
        """Test partial registry models are valid."""
        # Minimal fields
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.pricing is None
        assert model.context_length is None

    def test_unique_id_normalization(self) -> None:
        """Test unique ID generation and normalization."""
        # Uppercase provider and model
        model = RegistryModel(provider="OpenAI", model_name="GPT-4")
        assert model.unique_id() == "openai:gpt-4"  # Normalized to lowercase

        # Mixed case
        model = RegistryModel(provider="Anthropic", model_name="Claude-3-Sonnet")
        assert model.unique_id() == "anthropic:claude-3-sonnet"

    def test_unique_id_with_normalized_model_names(self) -> None:
        """Test unique_id with normalized model names (no slashes in schema)."""
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.unique_id() == "openai:gpt-4"

        model = RegistryModel(provider="anthropic", model_name="claude-3-sonnet")
        assert model.unique_id() == "anthropic:claude-3-sonnet"

    def test_unique_id_error_cases(self) -> None:
        """Test unique ID generation error handling."""
        # Missing provider
        model = RegistryModel(provider="", model_name="gpt-4")
        with pytest.raises(RegistryError, match="missing provider"):
            model.unique_id()

        # Missing model name
        model = RegistryModel(provider="openai", model_name="")
        with pytest.raises(RegistryError, match="missing model_name"):
            model.unique_id()

    def test_average_price_edge_cases(self) -> None:
        """Test average_price calculation edge cases."""
        # Normal case
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "0.00003", "completion_cost": "0.00006"},
        )
        assert model.average_price() == pytest.approx(0.000045)

        # Zero costs
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "0", "completion_cost": "0"},
        )
        assert model.average_price() is None

        # Missing pricing
        model = RegistryModel(provider="openai", model_name="gpt-4")
        assert model.average_price() is None

        # Invalid pricing (non-numeric)
        model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            pricing={"prompt_cost": "invalid", "completion_cost": "0.00006"},
        )
        assert model.average_price() is None


class TestModelSelectionRequestValidation:
    """Test ModelSelectionRequest validation edge cases."""

    def test_prompt_whitespace_handling(self) -> None:
        """Test prompt whitespace is trimmed."""
        request = ModelSelectionRequest(prompt="  test prompt  ")
        assert request.prompt == "test prompt"

    def test_cost_bias_boundary_values(self) -> None:
        """Test cost_bias boundary validation."""
        # Valid boundaries
        ModelSelectionRequest(prompt="test", cost_bias=0.0)
        ModelSelectionRequest(prompt="test", cost_bias=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=-0.1)

        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=1.1)

    def test_empty_models_list(self) -> None:
        """Test empty models list is valid."""
        request = ModelSelectionRequest(prompt="test", models=[])
        assert request.models == []

    def test_none_models_list(self) -> None:
        """Test None models list is valid."""
        request = ModelSelectionRequest(prompt="test", models=None)
        assert request.models is None


class TestModelSelectionResponseValidation:
    """Test ModelSelectionResponse validation edge cases."""

    def test_whitespace_trimming(self) -> None:
        """Test model_id whitespace is trimmed."""
        response = ModelSelectionResponse(
            model_id="  openai:gpt-4  ",
            alternatives=[],
        )
        assert response.model_id == "openai:gpt-4"

    def test_empty_alternatives(self) -> None:
        """Test empty alternatives list is valid."""
        response = ModelSelectionResponse(
            model_id="openai:gpt-4",
            alternatives=[],
        )
        assert response.alternatives == []


class TestAlternativeValidation:
    """Test Alternative validation edge cases."""

    def test_whitespace_trimming(self) -> None:
        """Test model_id whitespace is trimmed."""
        alt = Alternative(model_id="  openai:gpt-4  ")
        assert alt.model_id == "openai:gpt-4"
