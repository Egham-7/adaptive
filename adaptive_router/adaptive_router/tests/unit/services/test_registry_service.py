"""Tests for the HTTP-backed model registry cache."""

from __future__ import annotations

import pytest

from adaptive_router.registry.client import RegistryModel
from adaptive_router.registry.registry import ModelRegistry


class _FakeRegistryClient:
    def __init__(self, models: list[RegistryModel]) -> None:
        self._models = models
        self.calls = 0

    def list_models(self, *_, **__) -> list[RegistryModel]:
        self.calls += 1
        return list(self._models)


@pytest.fixture
def registry_models() -> list[RegistryModel]:
    return [
        RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "openrouter_id": "openai/gpt-4",
                "pricing": {"prompt": "0.00001", "completion": "0.00002"},
                "context_length": 128_000,
                "supported_parameters": ["tools", "max_tokens"],
            }
        ),
        RegistryModel.model_validate(
            {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "openrouter_id": "anthropic/claude-3-sonnet",
                "pricing": {"prompt": "0.00002"},
                "context_length": 200_000,
                "supported_parameters": ["temperature"],
            }
        ),
    ]


@pytest.fixture
def fake_client(registry_models: list[RegistryModel]) -> _FakeRegistryClient:
    return _FakeRegistryClient(registry_models)


@pytest.fixture
def model_registry(fake_client: _FakeRegistryClient) -> ModelRegistry:
    return ModelRegistry(fake_client)


def test_refresh_fetches_models(fake_client: _FakeRegistryClient) -> None:
    registry = ModelRegistry(fake_client)
    assert fake_client.calls == 1
    assert len(registry.list_models()) == 2


def test_get_by_unique_id(model_registry: ModelRegistry) -> None:
    model = model_registry.get("openai:gpt-4")
    assert model is not None
    assert model.provider == "openai"


def test_get_by_name(model_registry: ModelRegistry) -> None:
    models = model_registry.get_by_name("gpt-4")
    assert len(models) == 1
    assert models[0].openrouter_id == "openai/gpt-4"


def test_providers_for_model(model_registry: ModelRegistry) -> None:
    providers = model_registry.providers_for_model("claude-3-sonnet")
    assert providers == {"anthropic"}


def test_filter_supports_function_calling(model_registry: ModelRegistry) -> None:
    results = model_registry.filter(requires_function_calling=True)
    assert len(results) == 1
    assert results[0].provider == "openai"


def test_refresh_replaces_cache(fake_client: _FakeRegistryClient) -> None:
    registry = ModelRegistry(fake_client)

    replacement = RegistryModel.model_validate(
        {
            "provider": "mistral",
            "model_name": "mistral-large",
            "openrouter_id": "mistral/mistral-large",
        }
    )

    fake_client._models = [replacement]
    registry.refresh()

    assert registry.get("mistral:mistral-large") is not None
    assert registry.get("openai:gpt-4") is None
