"""Integration tests for FastAPI endpoints and library usage.

This module tests deployment modes:
1. FastAPI HTTP server (localhost:8000) - for local/production deployment
2. Library usage - direct Python imports (no server required)

Test Categories:
- FastAPI endpoint functionality
- Request/response validation
- Error handling
- Library direct usage
"""

import pytest


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
class TestLibraryUsage:
    """Test using model_router as a library (no server required)."""

    @pytest.mark.skip(reason="PromptClassifier not in package exports")
    def test_direct_classifier_usage(self) -> None:
        """Test using the classifier directly as a library.

        This test doesn't require any server - just imports and uses the library.
        Note: First run will download the ML model (~500MB).
        """
        from adaptive_router import PromptClassifier

        classifier = PromptClassifier()
        result = classifier.classify_prompt("Write a Python sorting function")

        assert "task_type_1" in result
        assert "prompt_complexity_score" in result
        assert isinstance(result["task_type_1"], str)
        assert isinstance(result["prompt_complexity_score"], float)
        assert 0.0 <= result["prompt_complexity_score"] <= 1.0

    @pytest.mark.skip(reason="No local profile file available for testing")
    def test_direct_router_usage(self) -> None:
        """Test using the router directly as a library."""
        from pathlib import Path
        import yaml
        from adaptive_router import ModelRouter, ModelSelectionRequest

        config_file = (
            Path(__file__).parent.parent.parent / "config" / "unirouter_models.yaml"
        )
        profile_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "global_profile.json"
        )

        with open(config_file) as f:
            config = yaml.safe_load(f)
            model_costs = {
                model["id"]: model["cost_per_1m_tokens"]
                for model in config.get("gpt5_models", [])
            }

        router = ModelRouter.from_local_file(
            profile_path=profile_path,
            model_costs=model_costs,
        )
        request = ModelSelectionRequest(
            prompt="Explain machine learning", cost_bias=0.5
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)
