"""Model fixtures for tests."""

import pytest

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)


@pytest.fixture
def sample_model_capability():
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
def anthropic_model_capability():
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
def sample_models_list(sample_model_capability, anthropic_model_capability):
    """List of sample models for testing."""
    return [sample_model_capability, anthropic_model_capability]


@pytest.fixture
def sample_model_selection_request():
    """Sample ModelSelectionRequest for testing."""
    return ModelSelectionRequest(
        prompt="Write a Python function to calculate factorial",
        cost_bias=0.5,
        user_id="test_user_123",
    )


@pytest.fixture
def sample_model_selection_response():
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
def sample_classification_result():
    """Sample ClassificationResult for testing."""
    return ClassificationResult(
        # Required fields
        task_type_1=["code", "analysis"],
        prompt_complexity_score=[0.75, 0.65],
        domain=["Programming", "Analytics"],
        # Optional fields
        task_type_2=["generation", "problem_solving"],
        task_type_prob=[0.75, 0.70],
        creativity_scope=[0.4, 0.3],
        reasoning=[0.8, 0.7],
        contextual_knowledge=[0.6, 0.5],
        domain_knowledge=[0.5, 0.6],
        number_of_few_shots=[0, 1],
        no_label_reason=[0.9, 0.85],
        constraint_ct=[0.3, 0.2],
    )


@pytest.fixture
def empty_classification_result():
    """Empty ClassificationResult for testing edge cases."""
    return ClassificationResult(
        # Required fields cannot be empty, use minimal valid values
        task_type_1=["Other"],
        prompt_complexity_score=[0.0],
        domain=["General"],
        # Optional fields can be empty
        task_type_2=[],
        task_type_prob=[],
        creativity_scope=[],
        reasoning=[],
        contextual_knowledge=[],
        domain_knowledge=[],
        number_of_few_shots=[],
        no_label_reason=[],
        constraint_ct=[],
    )


@pytest.fixture
def partial_model_capability():
    """Partial ModelCapability for testing partial specifications."""
    return ModelCapability(
        provider="openai",
        model_name="gpt-4",
        # Missing other fields to make it partial
    )


@pytest.fixture
def complex_model_capabilities():
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
