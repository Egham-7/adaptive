"""Tests for the HTTP-backed model registry cache."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.models import RegistryModel
from app.registry.models import ModelRegistry


@pytest.fixture
def registry_models() -> list[RegistryModel]:
    return [
        RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": "0.00001", "completion_cost": "0.00002"},
                "context_length": 128_000,
                "supported_parameters": ["tools", "max_tokens"],
            }
        ),
        RegistryModel.model_validate(
            {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "pricing": {"prompt_cost": "0.00002", "completion_cost": "0.00001"},
                "context_length": 200_000,
                "supported_parameters": ["temperature"],
            }
        ),
    ]


@pytest.fixture
def mock_client(registry_models: list[RegistryModel]) -> Mock:
    """Mock RegistryClient that returns the test models."""
    client = Mock()
    client.list_models.return_value = registry_models
    return client


@pytest.fixture
def model_registry(mock_client: Mock) -> ModelRegistry:
    return ModelRegistry(mock_client)


def test_refresh_fetches_models(mock_client: Mock) -> None:
    registry = ModelRegistry(mock_client)
    mock_client.list_models.assert_called_once()
    assert len(registry.list_models()) == 2


def test_get_by_unique_id(model_registry: ModelRegistry) -> None:
    model = model_registry.get("openai:gpt-4")
    assert model is not None
    assert model.provider == "openai"


def test_get_by_name(model_registry: ModelRegistry) -> None:
    models = model_registry.get_by_name("gpt-4")
    assert len(models) == 1
    assert models[0].provider == "openai"
    assert models[0].model_name == "gpt-4"


def test_providers_for_model(model_registry: ModelRegistry) -> None:
    providers = model_registry.providers_for_model("claude-3-sonnet")
    assert providers == {"anthropic"}


def test_filter_supports_function_calling(model_registry: ModelRegistry) -> None:
    results = model_registry.filter(requires_function_calling=True)
    assert len(results) == 1
    assert results[0].provider == "openai"


def test_refresh_replaces_cache(mock_client: Mock) -> None:
    registry = ModelRegistry(mock_client)

    replacement = RegistryModel.model_validate(
        {
            "provider": "mistral",
            "model_name": "mistral-large",
        }
    )

    mock_client.list_models.return_value = [replacement]
    registry.refresh()

    assert registry.get("mistral:mistral-large") is not None
    assert registry.get("openai:gpt-4") is None
