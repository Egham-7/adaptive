"""Test case sensitivity fix works correctly by using models without task_type restrictions."""

import os

import pytest
import requests


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skip integration tests in CI environment"
)
class TestCaseSensitivityFix:
    """Test case sensitivity handling is fixed."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_case_sensitivity_with_full_model(self, base_url):
        """Test case sensitivity with full model specs (no task type filtering)."""
        # Test different case variations with FULL model specs
        test_cases = [
            {
                "provider": "OPENAI",
                "model_name": "My-Custom-Model",
                "cost_per_1m_input_tokens": 5.0,
                "cost_per_1m_output_tokens": 10.0,
                "max_context_tokens": 8192,
                "supports_function_calling": True,
            },
            {
                "provider": "openai",
                "model_name": "my-custom-model",
                "cost_per_1m_input_tokens": 5.0,
                "cost_per_1m_output_tokens": 10.0,
                "max_context_tokens": 8192,
                "supports_function_calling": True,
            },
        ]

        for model_spec in test_cases:
            request_data = {
                "prompt": "Test case sensitivity",
                "models": [model_spec],
            }

            response = requests.post(
                f"{base_url}/predict", json=request_data, timeout=30
            )

            # Full custom models should work regardless of case
            assert response.status_code == 200, f"Failed for {model_spec}"
            result = response.json()

            # Should use the custom model with normalized case
            assert result["provider"].lower() == "openai"
            assert result["model"].lower() == "my-custom-model"

    def test_error_messages_are_specific(self, base_url):
        """Test that error messages include specific details."""
        request_data = {
            "prompt": "Test error messages",
            "models": [
                {"provider": "NonExistentProvider", "model_name": "FakeModel123"}
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200  # LitServe returns 200 even for errors
        result = response.json()

        # Should have error information
        assert "error" in result
        assert result["error"] == "ValueError"
        assert (
            "NonExistentProvider" in result["message"]
            or "FakeModel123" in result["message"]
        )

    def test_empty_array_error(self, base_url):
        """Test empty model array returns specific error."""
        request_data = {
            "prompt": "Test empty array",
            "models": [],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200  # LitServe returns 200 even for errors
        result = response.json()

        # Should have error information
        assert "error" in result
        assert result["error"] == "ValueError"
        assert "empty model array" in result["message"].lower()
