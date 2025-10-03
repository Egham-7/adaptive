"""Integration tests for FastAPI endpoints and Modal functions.

This module tests both deployment modes:
1. FastAPI HTTP server (localhost:8000) - for local development
2. Modal function calls - for serverless deployment

Test Categories:
- FastAPI endpoint functionality
- Modal function invocation
- Request/response validation
- Error handling
"""

import pytest

from adaptive_router.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)


@pytest.mark.integration
class TestFastAPIEndpoints:
    """Test FastAPI HTTP server endpoints (requires server running on localhost:8000)."""

    @pytest.mark.skip(reason="FastAPI server must be running manually for this test")
    def test_select_model_endpoint(self) -> None:
        """Test /select_model endpoint with FastAPI server.

        To run this test:
        1. Start the server: uv run model-router
        2. Run: pytest -m integration tests/integration/test_api_endpoints.py -k test_select_model_endpoint
        """
        import requests

        request_data = {
            "prompt": "Write a Python function to calculate factorial",
            "cost_bias": 0.5,
        }

        response = requests.post(
            "http://localhost:8000/select_model",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        assert "provider" in result
        assert "model" in result
        assert "alternatives" in result

    @pytest.mark.skip(reason="FastAPI server must be running manually for this test")
    def test_health_endpoint(self) -> None:
        """Test /health endpoint.

        To run this test:
        1. Start the server: uv run model-router
        2. Run: pytest -m integration tests/integration/test_api_endpoints.py -k test_health_endpoint
        """
        import requests

        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200

        result = response.json()
        assert "status" in result
        assert result["status"] == "healthy"


@pytest.mark.integration
class TestModalFunctions:
    """Test Modal function invocation (requires Modal deployment).

    These tests call Modal functions directly using the Modal SDK.
    They require the service to be deployed to Modal first.
    """

    @pytest.mark.skip(reason="Modal deployment required - run: modal deploy deploy.py")
    def test_select_model_function(self) -> None:
        """Test select_model Modal function.

        To run this test:
        1. Deploy to Modal: modal deploy deploy.py
        2. Run: pytest -m integration tests/integration/test_api_endpoints.py -k test_select_model_function
        """
        from modal import Function

        # Lookup the deployed function
        select_model = Function.lookup(
            "prompt-task-complexity-classifier", "select_model"
        )

        # Create request
        request = ModelSelectionRequest(
            prompt="Write a Python function to calculate factorial",
            cost_bias=0.5,
        )

        # Call the Modal function
        response = select_model.remote(request)

        # Validate response
        assert isinstance(response, ModelSelectionResponse)
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    @pytest.mark.skip(reason="Modal deployment required - run: modal deploy deploy.py")
    def test_select_model_with_specific_models(self) -> None:
        """Test select_model Modal function with specific models provided."""
        from modal import Function

        select_model = Function.lookup(
            "prompt-task-complexity-classifier", "select_model"
        )

        # Create request with specific models
        request = ModelSelectionRequest(
            prompt="Explain quantum computing",
            models=[
                ModelCapability(provider="openai", model_name="gpt-4"),
                ModelCapability(
                    provider="anthropic", model_name="claude-3-5-sonnet-20241022"
                ),
            ],
            cost_bias=0.3,
        )

        response = select_model.remote(request)

        assert isinstance(response, ModelSelectionResponse)
        assert response.provider in ["openai", "anthropic"]


@pytest.mark.integration
class TestLibraryUsage:
    """Test using model_router as a library (no server/Modal required)."""

    def test_direct_classifier_usage(self) -> None:
        """Test using the classifier directly as a library.

        This test doesn't require any server - just imports and uses the library.
        Note: First run will download the ML model (~500MB).
        """
        from model_router import PromptClassifier

        classifier = PromptClassifier()
        result = classifier.classify_prompt("Write a Python sorting function")

        assert "task_type_1" in result
        assert "prompt_complexity_score" in result
        assert isinstance(result["task_type_1"], str)
        assert isinstance(result["prompt_complexity_score"], float)
        assert 0.0 <= result["prompt_complexity_score"] <= 1.0

    def test_direct_router_usage(self) -> None:
        """Test using the router directly as a library."""
        from model_router import ModelRouter, ModelSelectionRequest

        router = ModelRouter()
        request = ModelSelectionRequest(
            prompt="Explain machine learning", cost_bias=0.5
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)
