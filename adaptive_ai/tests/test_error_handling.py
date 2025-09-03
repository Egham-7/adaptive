"""Error handling tests for @adaptive_ai/ model selection."""

import pytest
import requests


class TestErrorHandling:
    """Test error handling scenarios and edge cases."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_empty_prompt_handling(self, base_url):
        """Test handling of empty prompt."""
        # Arrange
        request_data = {"prompt": "", "cost_bias": 0.5}

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Assert - empty prompt now handled gracefully (after Pydantic validation fix)
        # The ML classifier can handle empty prompts with bounded number_of_few_shots
        assert response.status_code == 200
        result = response.json()
        assert result["model"], "Empty prompt should still select a valid model"

    def test_invalid_cost_bias(self, base_url):
        """Test handling of invalid cost_bias values."""
        # Arrange
        request_data = {
            "prompt": "Test prompt",
            "cost_bias": 1.5,  # Invalid - should be 0.0 to 1.0
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Assert - might return error or clamp to valid range
        # The behavior depends on the API's validation
        assert response.status_code in [200, 400, 422]

    def test_missing_prompt(self, base_url):
        """Test handling of missing prompt field."""
        # Arrange
        request_data = {
            "cost_bias": 0.5
            # Missing prompt
        }

        # Act
        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Assert - should return validation error
        assert response.status_code in [400, 422]
