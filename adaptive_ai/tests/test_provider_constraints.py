"""Provider constraint tests for @adaptive_ai/ model selection."""

import pytest
import requests


class TestProviderConstraints:
    """Test provider-specific routing and constraints."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_provider_only_constraint(self, base_url):
        """Test routing with provider-only constraint (no specific model)."""
        # Arrange - constrain to Anthropic only
        request_data = {
            "prompt": "Write a Python function to sort a list",
            "models": [
                {
                    "provider": "ANTHROPIC"
                    # No model_name - let system pick best Anthropic model
                }
            ],
            "cost_bias": 0.5,
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        result = response.json()

        # Should use an Anthropic model
        assert result["provider"].upper() == "ANTHROPIC"
        assert "claude" in result["model"].lower()  # Should be a Claude model

        # Log for debugging
        print(
            f"Anthropic-only constraint routed to: {result['provider']}/{result['model']}"
        )
        print(f"Alternatives: {result.get('alternatives', [])}")

    def test_openai_only_constraint(self, base_url):
        """Test routing with OpenAI-only constraint."""
        # Arrange - constrain to OpenAI only
        request_data = {
            "prompt": "Solve this math problem: 2x + 5 = 17",
            "models": [
                {
                    "provider": "OPENAI"
                    # No model_name - let system pick best OpenAI model
                }
            ],
            "cost_bias": 0.7,  # Prefer quality
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        result = response.json()

        # Should use an OpenAI model
        assert result["provider"].upper() == "OPENAI"
        assert result["model"] in [
            "gpt-5",
            "gpt-4o",
            "o3",
            "o3-mini",
            "gpt-4.1",
            "gpt-3.5-turbo",
        ]

        # Log for debugging
        print(
            f"OpenAI-only constraint routed to: {result['provider']}/{result['model']}"
        )
