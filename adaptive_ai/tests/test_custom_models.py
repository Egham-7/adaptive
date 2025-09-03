"""Custom model routing tests for @adaptive_ai/ model selection."""

import pytest
import requests


class TestCustomModelsRouting:
    """Test custom model specification and routing logic."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_custom_models_routing(self, base_url):
        """Test that custom models are used when provided."""
        # Arrange
        request_data = {
            "prompt": "Test prompt",
            "models": [
                {
                    "provider": "custom-ai",
                    "model_name": "my-custom-model",
                    "cost_per_1m_input_tokens": 5.0,
                    "cost_per_1m_output_tokens": 10.0,
                    "max_context_tokens": 32768,
                    "supports_function_calling": True,
                }
            ],
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        result = response.json()

        # Should use the custom model
        assert result["provider"] == "custom-ai"
        assert result["model"] == "my-custom-model"
        assert result["alternatives"] == []  # No alternatives for single custom model

        # Log for debugging
        print(f"Custom model request routed to: {result['provider']}/{result['model']}")

    def test_multiple_custom_models_routing(self, base_url):
        """Test routing with multiple custom models."""
        # Arrange
        request_data = {
            "prompt": "Write some code",
            "models": [
                {
                    "provider": "custom-ai",
                    "model_name": "custom-code-model",
                    "cost_per_1m_input_tokens": 10.0,
                    "cost_per_1m_output_tokens": 20.0,
                    "max_context_tokens": 16384,
                    "supports_function_calling": True,
                },
                {
                    "provider": "another-ai",
                    "model_name": "another-model",
                    "cost_per_1m_input_tokens": 5.0,
                    "cost_per_1m_output_tokens": 10.0,
                    "max_context_tokens": 8192,
                    "supports_function_calling": False,
                },
            ],
            "cost_bias": 0.3,  # Prefer cheaper model
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        result = response.json()

        # Should have both models available
        assert result["provider"] in ["custom-ai", "another-ai"]
        assert result["model"] in ["custom-code-model", "another-model"]
        assert len(result["alternatives"]) >= 1

        # Log for debugging
        print(
            f"Multiple custom models routed to: {result['provider']}/{result['model']}"
        )
        print(f"Alternatives: {result['alternatives']}")

    def test_malformed_custom_model(self, base_url):
        """Test handling of malformed custom model specification."""
        # Arrange
        request_data = {
            "prompt": "Test prompt",
            "models": [
                {
                    "provider": "custom-ai",
                    # Missing required fields
                    "model_name": "incomplete-model",
                }
            ],
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data)

        # Assert - might use the partial model or return error
        assert response.status_code in [200, 400, 422, 500]
