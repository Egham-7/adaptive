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
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": "0.00001", "completion_cost": "0.00002"},
                "context_length": 128_000,
                "supported_parameters": [
                    {"parameter_name": "tools"},
                    {"parameter_name": "max_tokens"},
                ],
            }
        ),
        RegistryModel.model_validate(
            {
                "author": "anthropic",
                "model_name": "claude-3-sonnet",
                "pricing": {"prompt_cost": "0.00002", "completion_cost": "0.00001"},
                "context_length": 200_000,
                "supported_parameters": [{"parameter_name": "temperature"}],
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
def model_registry(
    mock_client: Mock, registry_models: list[RegistryModel]
) -> ModelRegistry:
    return ModelRegistry(mock_client, registry_models)


def test_list_models(model_registry: ModelRegistry) -> None:
    models = model_registry.list_models()
    assert len(models) == 2


def test_get_by_unique_id(model_registry: ModelRegistry) -> None:
    model = model_registry.get("openai/gpt-4")
    assert model is not None
    assert model.author == "openai"


def test_get_by_name(model_registry: ModelRegistry) -> None:
    models = model_registry.get_by_name("gpt-4")
    assert len(models) == 1
    assert models[0].author == "openai"
    assert models[0].model_name == "gpt-4"


def test_authors_for_model(model_registry: ModelRegistry) -> None:
    authors = model_registry.authors_for_model("claude-3-sonnet")
    assert authors == {"anthropic"}
