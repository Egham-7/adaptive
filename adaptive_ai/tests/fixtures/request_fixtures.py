"""Request and response fixtures for integration tests."""

import pytest


@pytest.fixture
def base_url():
    """Base URL for API integration tests."""
    return "http://localhost:8000"


@pytest.fixture
def sample_predict_request():
    """Sample prediction request data."""
    return {
        "prompt": "Write a Python function to calculate factorial",
        "cost_bias": 0.5,
    }


@pytest.fixture
def code_generation_request():
    """Request for code generation tasks."""
    return {
        "prompt": "Create a Python class that implements a binary search tree with insert, delete, and search methods",
        "cost_bias": 0.3,  # Prefer performance
    }


@pytest.fixture
def analysis_request():
    """Request for analysis tasks."""
    return {
        "prompt": "Analyze the economic implications of remote work adoption in the technology sector",
        "cost_bias": 0.7,  # Prefer quality
    }


@pytest.fixture
def creative_writing_request():
    """Request for creative writing tasks."""
    return {
        "prompt": "Write a short story about a robot that discovers emotions",
        "cost_bias": 0.8,  # Prefer quality for creativity
    }


@pytest.fixture
def math_problem_request():
    """Request for math problem solving."""
    return {
        "prompt": "Solve this system of equations: 2x + 3y = 12, 4x - y = 5. Show your work.",
        "cost_bias": 0.4,  # Balance cost and accuracy
    }


@pytest.fixture
def custom_models_request():
    """Request with custom model specifications."""
    return {
        "prompt": "Explain quantum computing",
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
                "max_context_tokens": 128000,
                "supports_function_calling": True,
            },
            {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "cost_per_1m_input_tokens": 15.0,
                "cost_per_1m_output_tokens": 75.0,
                "max_context_tokens": 200000,
                "supports_function_calling": False,
            },
        ],
        "cost_bias": 0.6,
    }


@pytest.fixture
def large_context_request():
    """Request requiring large context handling."""
    long_prompt = "Analyze this document: " + ("This is a long document. " * 1000)
    return {"prompt": long_prompt, "cost_bias": 0.5}


@pytest.fixture
def multiple_requests_batch():
    """Batch of requests for concurrent testing."""
    return [
        {"prompt": f"Request {i}: Simple calculation {i} + {i}", "cost_bias": 0.5}
        for i in range(5)
    ]
