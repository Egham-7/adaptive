"""ML model and service fixtures for testing."""

from unittest.mock import Mock

import pytest

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import TaskType


@pytest.fixture
def mock_classification_model():
    """Mock ML classification model."""
    mock_model = Mock()
    mock_model.predict.return_value = [["code", "analysis"]]
    mock_model.predict_proba.return_value = [[0.8, 0.2, 0.0, 0.0, 0.0, 0.0]]
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for text processing."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
    mock_tokenizer.decode.return_value = "decoded text"
    return mock_tokenizer


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer model."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Mock embeddings
    return mock_model


@pytest.fixture
def mock_prompt_classifier():
    """Mock prompt classifier service."""
    classifier = Mock()
    classifier.classify_prompts.return_value = [
        ClassificationResult(
            # Required fields
            task_type=["code"],
            complexity_score=[0.75],
            domain=["Programming"],
            # Optional fields
            task_type_1=["code"],
            task_type_2=["generation"],
            task_type_prob=[0.85],
            creativity_scope=[0.4],
            reasoning=[0.8],
            contextual_knowledge=[0.6],
            prompt_complexity_score=[0.75],
            domain_knowledge=[0.5],
            number_of_few_shots=[0],
            no_label_reason=[0.9],
            constraint_ct=[0.3],
        ),
        ClassificationResult(
            # Required fields
            task_type=["analysis"],
            complexity_score=[0.65],
            domain=["Analytics"],
            # Optional fields
            task_type_1=["analysis"],
            task_type_2=["problem_solving"],
            task_type_prob=[0.72],
            creativity_scope=[0.6],
            reasoning=[0.9],
            contextual_knowledge=[0.7],
            prompt_complexity_score=[0.65],
            domain_knowledge=[0.4],
            number_of_few_shots=[0],
            no_label_reason=[0.8],
            constraint_ct=[0.2],
        ),
    ]
    return classifier


@pytest.fixture
def mock_model_router():
    """Mock model router service."""
    router = Mock()
    router.select_models.return_value = [
        ModelCapability(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=128000,
            supports_function_calling=True,
            task_type=TaskType.CODE_GENERATION,
        ),
        ModelCapability(
            provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=75.0,
            max_context_tokens=200000,
            supports_function_calling=False,
            task_type=TaskType.TEXT_GENERATION,
        ),
    ]
    return router


@pytest.fixture
def mock_model_registry():
    """Mock model registry service."""
    registry = Mock()
    registry.get_models_by_task.return_value = [
        ModelCapability(
            provider="openai", model_name="gpt-4", task_type=TaskType.CODE_GENERATION
        ),
        ModelCapability(
            provider="anthropic",
            model_name="claude-3-sonnet",
            task_type=TaskType.CODE_GENERATION,
        ),
    ]
    registry.get_all_models.return_value = {
        "gpt-4": {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
        },
        "claude-3-sonnet": {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet",
            "cost_per_1m_input_tokens": 15.0,
        },
    }
    return registry


@pytest.fixture
def mock_yaml_model_db():
    """Mock YAML model database."""
    mock_db = Mock()
    mock_db.get_model.return_value = {
        "provider": "openai",
        "model_name": "gpt-4",
        "cost_per_1m_input_tokens": 30.0,
        "cost_per_1m_output_tokens": 60.0,
        "max_context_tokens": 128000,
        "supports_function_calling": True,
    }
    mock_db.get_all_models.return_value = {
        "gpt-4": {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
        },
        "claude-3-sonnet": {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "cost_per_1m_input_tokens": 15.0,
        },
    }
    return mock_db


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return {
        "code_prompt": [0.1, 0.2, 0.8, 0.1, 0.0],
        "analysis_prompt": [0.0, 0.1, 0.2, 0.7, 0.8],
        "creative_prompt": [0.8, 0.7, 0.1, 0.0, 0.2],
        "math_prompt": [0.1, 0.0, 0.2, 0.1, 0.9],
    }


@pytest.fixture
def classification_test_prompts():
    """Sample prompts for classification testing."""
    return {
        "code": "Write a Python function to sort a list using quicksort algorithm",
        "analysis": "Analyze the market trends in renewable energy sector for 2024",
        "creative": "Write a short poem about artificial intelligence and human creativity",
        "math": "Solve the differential equation dy/dx = 2x + 3, given y(0) = 5",
        "chat": "Hello, how are you doing today?",
        "other": "What's the weather like?",
    }
