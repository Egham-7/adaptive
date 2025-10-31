"""Model fixtures for tests."""

import pytest

from adaptive_router.models.api import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)


@pytest.fixture
def sample_model_capability() -> ModelCapability:
    """Sample ModelCapability instance for testing."""
    return ModelCapability(
        provider="openai",
        model_name="gpt-4",
        cost_per_1m_input_tokens=30.0,
        cost_per_1m_output_tokens=60.0,
        max_context_tokens=128000,
        supports_function_calling=True,
        task_type="Code Generation",
        complexity="medium",
        description="GPT-4 model for complex tasks",
    )


@pytest.fixture
def anthropic_model_capability() -> ModelCapability:
    """Anthropic model capability for testing."""
    return ModelCapability(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        cost_per_1m_input_tokens=15.0,
        cost_per_1m_output_tokens=75.0,
        max_context_tokens=200000,
        supports_function_calling=False,
        task_type="Text Generation",
        complexity="high",
        description="Claude 3 Sonnet for analysis tasks",
    )


@pytest.fixture
def sample_models_list(
    sample_model_capability: ModelCapability,
    anthropic_model_capability: ModelCapability,
) -> list[ModelCapability]:
    """List of sample models for testing."""
    return [sample_model_capability, anthropic_model_capability]


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
def partial_model_capability() -> ModelCapability:
    """Partial ModelCapability for testing partial specifications."""
    return ModelCapability(
        provider="openai",
        model_name="gpt-4",
        # Missing other fields to make it partial
    )


@pytest.fixture
def complex_model_capabilities() -> list[ModelCapability]:
    """Various model capabilities for testing complex scenarios."""
    return [
        ModelCapability(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=128000,
            supports_function_calling=True,
            task_type="Code Generation",
            complexity="hard",
        ),
        ModelCapability(
            provider="anthropic",
            model_name="claude-3-haiku",
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.25,
            max_context_tokens=200000,
            supports_function_calling=False,
            task_type="Chatbot",
            complexity="easy",
        ),
        ModelCapability(
            provider="deepseek",
            model_name="deepseek-chat",
            cost_per_1m_input_tokens=0.14,
            cost_per_1m_output_tokens=0.28,
            max_context_tokens=64000,
            supports_function_calling=True,
            task_type="Code Generation",
            complexity="medium",
        ),
    ]
