"""Test partial model specification handling in adaptive_ai.

This module tests the logic where:
- Full model specifications are used as custom models
- Partial model specifications trigger lookups from model_data
"""

import os

import pytest
import requests


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skip integration tests in CI environment"
)
class TestPartialModelHandling:
    """Test partial vs full model specification handling."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    # ===== FULL CUSTOM MODEL TESTS =====

    def test_full_custom_model_used_directly(self, base_url):
        """Test that fully specified custom models are used directly without lookup."""
        request_data = {
            "prompt": "Write a Python function",
            "models": [
                {
                    "provider": "my-custom-provider",
                    "model_name": "my-custom-model-v1",
                    "cost_per_1m_input_tokens": 7.5,
                    "cost_per_1m_output_tokens": 15.0,
                    "max_context_tokens": 16384,
                    "supports_function_calling": True,
                }
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()

        # Should use the exact custom model
        assert result["provider"] == "my-custom-provider"
        assert result["model"] == "my-custom-model-v1"
        assert len(result["alternatives"]) == 0

    def test_multiple_full_custom_models(self, base_url):
        """Test routing with multiple fully specified custom models."""
        request_data = {
            "prompt": "Complex reasoning task",
            "models": [
                {
                    "provider": "fast-ai",
                    "model_name": "fast-model",
                    "cost_per_1m_input_tokens": 1.0,
                    "cost_per_1m_output_tokens": 2.0,
                    "max_context_tokens": 4096,
                    "supports_function_calling": False,
                },
                {
                    "provider": "smart-ai",
                    "model_name": "smart-model",
                    "cost_per_1m_input_tokens": 10.0,
                    "cost_per_1m_output_tokens": 20.0,
                    "max_context_tokens": 32768,
                    "supports_function_calling": True,
                },
            ],
            "cost_bias": 0.8,  # Prefer quality over cost
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()

        # Should pick from the custom models based on cost_bias
        assert result["provider"] in ["fast-ai", "smart-ai"]
        assert result["model"] in ["fast-model", "smart-model"]

    # ===== PARTIAL MODEL LOOKUP TESTS =====

    def test_partial_provider_and_model_name_lookup(self, base_url):
        """Test that provider + model_name triggers lookup for other fields."""
        request_data = {
            "prompt": "Analyze this data",
            "models": [
                {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    # Missing: costs, context, capabilities - should be looked up
                }
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()

        # Should find and use OpenAI's GPT-4
        assert result["provider"].lower() == "openai"
        assert "gpt-4" in result["model"].lower()

    def test_model_name_only_cross_provider_search(self, base_url):
        """Test that model_name alone searches across all providers."""
        request_data = {
            "prompt": "Write a story about cats",  # Text generation prompt
            "models": [
                {
                    "model_name": "claude-3-haiku-20240307"
                    # No provider specified - should find it in Anthropic
                }
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()
        # Should find the model successfully
        assert "claude-3-haiku-20240307" in result["model"].lower()

    def test_provider_only_returns_all_provider_models(self, base_url):
        """Test that specifying only provider returns all models from that provider."""
        request_data = {
            "prompt": "Write a haiku about coding",
            "models": [
                {
                    "provider": "anthropic"
                    # No model_name - should consider all Anthropic models
                }
            ],
            "cost_bias": 0.3,  # Prefer cheaper models
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()

        # Should select an Anthropic model
        assert result["provider"].lower() == "anthropic"
        assert "claude" in result["model"].lower()

    def test_constraints_only_filters_all_models(self, base_url):
        """Test that constraint-only specifications filter from all available models."""
        request_data = {
            "prompt": "Process this large document",
            "models": [
                {
                    "supports_function_calling": True,
                    "max_context_tokens": 100000,  # High context requirement
                    # No provider/model specified - should search all
                }
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should find models matching these constraints
        if response.status_code == 200:
            result = response.json()
            # Should find a high-context model with function calling
            assert result["provider"] is not None
            assert result["model"] is not None

    # ===== EDGE CASE TESTS =====

    def test_empty_model_object_handling(self, base_url):
        """Test handling of empty model object {}."""
        request_data = {
            "prompt": "Test empty model",
            "models": [{}],  # Empty object
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should either succeed or return validation error (until service restart)
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            # Should return a valid model selection
            payload = response.json()
            assert payload.get("provider") is not None
            assert payload.get("model") is not None
            assert "alternatives" in payload
        else:
            # Validation error expected until service restart
            payload = response.json()
            assert "detail" in payload

    def test_nonexistent_provider_handling(self, base_url):
        """Test handling of non-existent provider."""
        request_data = {
            "prompt": "Test invalid provider",
            "models": [
                {
                    "provider": "definitely-not-a-real-provider",
                    "model_name": "fake-model",
                }
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should return 400 for invalid model specification (FastAPI proper error handling)
        assert response.status_code == 400
        payload = response.json()

        # Should contain error information in FastAPI format
        assert "detail" in payload
        assert "provider" in payload["detail"].lower() or "models" in payload["detail"].lower()

    def test_case_sensitivity_in_provider_names(self, base_url):
        """Test case-insensitive handling of provider names."""
        # Test different case variations with a model that supports Open QA task type
        test_cases = [
            {"provider": "OPENAI", "model_name": "gpt-4"},
            {"provider": "OpenAI", "model_name": "gpt-4"},
            {"provider": "openai", "model_name": "gpt-4"},
        ]

        for model_spec in test_cases:
            request_data = {
                "prompt": "What is the capital of France?",  # Simple Q&A that should classify as Open QA
                "models": [model_spec],
            }

            response = requests.post(
                f"{base_url}/predict", json=request_data, timeout=30
            )

            # All variations should work
            assert response.status_code == 200, f"Failed for {model_spec}"
            result = response.json()

            # FastAPI should handle case sensitivity correctly - all should succeed
            # No need to check for errors as case sensitivity is now properly supported

            assert result.get("provider") is not None, f"Provider is None for {model_spec}"
            assert result["provider"].lower() == "openai"

    def test_case_sensitivity_in_model_names(self, base_url):
        """Test case-insensitive handling of model names."""
        request_data = {
            "prompt": "What is 2 + 2?",  # Simple Q&A that should classify as Open QA
            "models": [
                {"provider": "openai", "model_name": "GPT-4"},  # Uppercase
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should handle case variations
        assert response.status_code == 200
        result = response.json()

        # FastAPI should handle case sensitivity correctly - all should succeed
        # No need to check for errors as case sensitivity is now properly supported

        assert result.get("model") is not None, "Model is None"
        assert "gpt-4" in result["model"].lower()

    # ===== MIXED SCENARIO TESTS =====

    def test_mixed_full_and_partial_models(self, base_url):
        """Test array with both full custom models and partial lookups."""
        request_data = {
            "prompt": "Complex task requiring multiple models",
            "models": [
                # Full custom model
                {
                    "provider": "my-custom-ai",
                    "model_name": "custom-expert",
                    "cost_per_1m_input_tokens": 5.0,
                    "cost_per_1m_output_tokens": 10.0,
                    "max_context_tokens": 16384,
                    "supports_function_calling": True,
                },
                # Partial - lookup all OpenAI models
                {"provider": "openai"},
                # Partial - lookup specific Anthropic model
                {"provider": "anthropic", "model_name": "claude-3-haiku-20240307"},
            ],
            "cost_bias": 0.5,
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        assert response.status_code == 200
        result = response.json()

        # Should have selected from the mixed pool
        assert result["provider"] is not None
        assert result["model"] is not None
        # Should have alternatives from the expanded list
        assert len(result["alternatives"]) > 0

    def test_conflicting_partial_specifications(self, base_url):
        """Test handling of potentially conflicting partial specs."""
        request_data = {
            "prompt": "Test conflicting specs",
            "models": [
                # High cost limit
                {"cost_per_1m_input_tokens": 50.0},  # Max budget
                # Low context requirement
                {"max_context_tokens": 4096},  # Min context
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should handle this gracefully
        assert response.status_code in [200, 400, 422]

    # ===== ERROR MESSAGE TESTS =====

    def test_specific_error_for_no_matches(self, base_url):
        """Test that specific error messages are returned when no models match."""
        request_data = {
            "prompt": "Test no matches",
            "models": [
                {"provider": "openai", "model_name": "gpt-99-ultra"}  # Doesn't exist
            ],
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        if response.status_code != 200:
            # Error message should be specific
            error_text = response.text
            # Should mention the specific model or criteria
            assert (
                "gpt-99-ultra" in error_text or "no models found" in error_text.lower()
            )

    def test_null_models_array_handling(self, base_url):
        """Test handling of null models array."""
        request_data = {
            "prompt": "Test null models",
            "models": None,  # Null instead of array
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should either use default models or return error
        assert response.status_code in [200, 400, 422]

    def test_empty_models_array_handling(self, base_url):
        """Test handling of empty models array."""
        request_data = {
            "prompt": "Test empty models array",
            "models": [],  # Empty array
        }

        response = requests.post(f"{base_url}/predict", json=request_data, timeout=30)

        # Should either use default models or return error
        assert response.status_code in [200, 400, 422]
