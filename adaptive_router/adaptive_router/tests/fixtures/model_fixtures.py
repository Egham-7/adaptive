"""Model fixtures for tests."""

import pytest

from adaptive_router.models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.registry import RegistryModel


@pytest.fixture
def sample_registry_model() -> RegistryModel:
    """Sample RegistryModel instance for testing."""
    return RegistryModel(
        provider="openai",
        model_name="gpt-4",
        description="GPT-4 model for complex tasks",
        context_length=128000,
        pricing={"prompt": "0.00003", "completion": "0.00006"},
    )


@pytest.fixture
def anthropic_registry_model() -> RegistryModel:
    """Anthropic registry model for testing."""
    return RegistryModel(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        description="Claude 3 Sonnet for analysis tasks",
        context_length=200000,
        pricing={"prompt": "0.000015", "completion": "0.000075"},
    )


@pytest.fixture
def sample_models_list(
    sample_registry_model: RegistryModel,
    anthropic_registry_model: RegistryModel,
) -> list[RegistryModel]:
    """List of sample models for testing."""
    return [sample_registry_model, anthropic_registry_model]


@pytest.fixture
def sample_model_selection_request() -> ModelSelectionRequest:
    """Sample ModelSelectionRequest for testing."""
    return ModelSelectionRequest(
        prompt="Write a Python function to calculate factorial",
        cost_bias=0.5,
        user_id="test_user_123",
    )


@pytest.fixture
def sample_model_selection_response() -> ModelSelectionResponse:
    """Sample ModelSelectionResponse for testing."""
    return ModelSelectionResponse(
        provider="openai",
        model="gpt-4",
        alternatives=[
            Alternative(provider="anthropic", model="claude-3-sonnet"),
            Alternative(provider="openai", model="gpt-3.5-turbo"),
        ],
    )


@pytest.fixture
def partial_registry_model() -> RegistryModel:
    """Partial RegistryModel for testing partial specifications."""
    return RegistryModel(
        provider="openai",
        model_name="gpt-4",
        # Missing optional fields like pricing, context_length
    )


@pytest.fixture
def complex_registry_models() -> list[RegistryModel]:
    """Various registry models for testing complex scenarios."""
    return [
        RegistryModel(
            provider="openai",
            model_name="gpt-4",
            context_length=128000,
            pricing={"prompt": "0.00003", "completion": "0.00006"},
            description="GPT-4 for complex tasks",
        ),
        RegistryModel(
            provider="anthropic",
            model_name="claude-3-haiku",
            context_length=200000,
            pricing={"prompt": "0.00000025", "completion": "0.00000125"},
            description="Claude 3 Haiku",
        ),
        RegistryModel(
            provider="deepseek",
            model_name="deepseek-chat",
            context_length=64000,
            pricing={"prompt": "0.00000014", "completion": "0.00000028"},
            description="DeepSeek Chat",
        ),
    ]
