"""Routing consistency tests for @adaptive_ai/ model selection."""

import pytest
import requests


class TestRoutingConsistency:
    """Test that routing is consistent and logical."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_same_prompt_consistent_routing(self, base_url):
        """Test that the same prompt gets consistent routing."""
        # Arrange
        request_data = {
            "prompt": "Write a function to calculate fibonacci numbers",
            "cost_bias": 0.5,
        }

        # Act - make multiple requests
        responses = []
        for _ in range(3):
            response = requests.post(f"{base_url}/predict", json=request_data)
            assert response.status_code == 200
            responses.append(response.json())

        # Assert - should get the same model each time
        models = [(r["provider"], r["model"]) for r in responses]
        assert all(m == models[0] for m in models), f"Inconsistent routing: {models}"

        print(f"Consistent routing verified: {models[0]}")
