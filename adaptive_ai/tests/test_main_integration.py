"""Integration tests for the main adaptive-ai service.

These tests require the service to be running:
    uv run adaptive-ai

Run these tests with:
    uv run pytest tests/test_main_integration.py -v
"""

import json
import time
from typing import Any, Dict, List

import pytest  # type: ignore
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # type: ignore


class TestAdaptiveAIIntegration:
    """End-to-end integration tests for the adaptive-ai service."""

    @pytest.fixture(scope="class")
    def api_client(self):
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @pytest.fixture(scope="class")
    def base_url(self):
        """Base URL for the adaptive-ai service."""
        return "http://localhost:8000"

    @pytest.fixture
    def health_check(self, api_client, base_url):
        """Verify service is running before tests."""
        try:
            response = api_client.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Service not healthy")
        except requests.exceptions.RequestException:
            pytest.skip("Service not running. Start with: uv run adaptive-ai")

    def test_service_health(self, api_client, base_url):
        """Test that the service health endpoint responds."""
        response = api_client.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        assert "status" in response.json()

    def test_single_request_basic(self, api_client, base_url, health_check):
        """Test a single basic chat completion request."""
        payload = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        assert "output" in result
        assert len(result["output"]) == 1
        assert "selected_models" in result["output"][0]
        assert "protocol_info" in result["output"][0]

    def test_batch_requests(self, api_client, base_url, health_check):
        """Test batch processing with multiple requests."""
        payloads = [
            {
                "messages": [{"role": "user", "content": "Write a Python function"}],
                "model": "gpt-4",
            },
            {
                "messages": [{"role": "user", "content": "Explain quantum physics"}],
                "model": "claude-3-sonnet",
            },
            {
                "messages": [{"role": "user", "content": "Translate to Spanish: Hello"}],
                "model": "gpt-3.5-turbo",
            },
        ]

        response = api_client.post(
            f"{base_url}/predict",
            json=payloads,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        assert "output" in result
        assert len(result["output"]) == 3
        
        # Verify each response has required fields
        for output in result["output"]:
            assert "selected_models" in output
            assert "protocol_info" in output
            assert "classification_result" in output

    def test_complex_code_generation(self, api_client, base_url, health_check):
        """Test with complex code generation prompt."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": """Write a Python class that implements a binary search tree with the following methods:
                    - insert(value)
                    - delete(value)
                    - search(value)
                    - inorder_traversal()
                    - get_height()
                    Include proper error handling and docstrings.""",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Should classify as code generation
        classification = output["classification_result"]
        assert "Code Generation" in classification.get("task_type_1", [])
        
        # Should have high complexity score
        assert classification.get("prompt_complexity_score", [0])[0] > 0.5

    def test_creative_writing_task(self, api_client, base_url, health_check):
        """Test with creative writing prompt."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short story about a time traveler who accidentally changes history by stepping on a butterfly.",
                }
            ],
            "model": "claude-3-sonnet",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Should classify as creative writing
        classification = output["classification_result"]
        assert "Creative Writing" in classification.get("task_type_1", [])
        
        # Should have high creativity score
        assert classification.get("creativity_scope", [0])[0] > 0.5

    def test_math_reasoning_task(self, api_client, base_url, health_check):
        """Test with mathematical reasoning prompt."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Solve this step by step: If a train travels 120 miles in 2 hours, and then speeds up to travel 180 miles in the next 2.5 hours, what is its average speed for the entire journey?",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Should have high reasoning score
        classification = output["classification_result"]
        assert classification.get("reasoning", [0])[0] > 0.5

    def test_with_conversation_history(self, api_client, base_url, health_check):
        """Test with multi-turn conversation history."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
                {"role": "user", "content": "Can you give me a specific example?"},
            ],
            "model": "gpt-3.5-turbo",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["output"]) == 1

    def test_with_tools_and_functions(self, api_client, base_url, health_check):
        """Test request with tools/functions specified."""
        payload = {
            "messages": [{"role": "user", "content": "What's the weather like?"}],
            "model": "gpt-4",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Protocol info should indicate tools are present
        assert output["protocol_info"].get("has_tools") is True

    def test_with_custom_model_config(self, api_client, base_url, health_check):
        """Test with custom model configuration."""
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "custom-model",
            "protocol_manager_config": {
                "models": [
                    {
                        "provider": "custom-provider",
                        "model_name": "custom-model-v1",
                        "cost_per_1m_input_tokens": 10.0,
                        "cost_per_1m_output_tokens": 20.0,
                        "max_context_tokens": 8192,
                        "supports_function_calling": False,
                    }
                ],
                "cost_bias": 0.8,
            },
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Should use custom model
        selected_models = output["selected_models"]
        assert any("custom-model" in model["model_name"] for model in selected_models)

    def test_edge_case_empty_message(self, api_client, base_url, health_check):
        """Test with empty message content."""
        payload = {
            "messages": [{"role": "user", "content": ""}],
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        # Should handle gracefully
        assert response.status_code in [200, 400]

    def test_edge_case_very_long_prompt(self, api_client, base_url, health_check):
        """Test with very long prompt exceeding typical limits."""
        # Generate a very long prompt
        long_content = "Please analyze this text: " + ("Lorem ipsum " * 10000)
        
        payload = {
            "messages": [{"role": "user", "content": long_content}],
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        output = result["output"][0]
        
        # Should handle high token count appropriately
        assert "selected_models" in output

    def test_edge_case_special_characters(self, api_client, base_url, health_check):
        """Test with special characters and unicode."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Translate: 你好世界 🌍 こんにちは ¿Cómo estás? #$%^&*()",
                }
            ],
            "model": "gpt-3.5-turbo",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["output"]) == 1

    def test_edge_case_malformed_request(self, api_client, base_url, health_check):
        """Test with malformed request structure."""
        payload = {
            "message": "This is wrong structure",  # Wrong field name
            "model": "gpt-4",
        }

        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )

        # Should return error
        assert response.status_code == 400 or response.status_code == 422

    def test_concurrent_requests(self, api_client, base_url, health_check):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request(content: str):
            payload = {
                "messages": [{"role": "user", "content": content}],
                "model": "gpt-4",
            }
            return api_client.post(
                f"{base_url}/predict",
                json=payload,
                timeout=30,
            )

        prompts = [
            "Write code",
            "Explain science",
            "Translate text",
            "Solve math",
            "Creative story",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, prompt) for prompt in prompts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        for response in results:
            assert response.status_code == 200
            assert "output" in response.json()

    def test_performance_response_time(self, api_client, base_url, health_check):
        """Test that response times are within acceptable limits."""
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }

        start_time = time.time()
        response = api_client.post(
            f"{base_url}/predict",
            json=payload,
            timeout=30,
        )
        end_time = time.time()

        assert response.status_code == 200
        
        # Response should be under 5 seconds for simple prompt
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response took {response_time:.2f} seconds"

    def test_model_selection_variety(self, api_client, base_url, health_check):
        """Test that different prompts select different models."""
        test_cases = [
            {
                "content": "Write a haiku",
                "expected_type": "Creative Writing",
            },
            {
                "content": "Debug this Python code: def factorial(n): return n * factorial(n-1)",
                "expected_type": "Code Generation",
            },
            {
                "content": "What is 2+2?",
                "expected_type": "Math",
            },
        ]

        models_selected = set()
        
        for test_case in test_cases:
            payload = {
                "messages": [{"role": "user", "content": test_case["content"]}],
                "model": "gpt-4",
            }
            
            response = api_client.post(
                f"{base_url}/predict",
                json=payload,
                timeout=30,
            )
            
            assert response.status_code == 200
            result = response.json()
            output = result["output"][0]
            
            # Collect selected models
            for model in output["selected_models"]:
                models_selected.add(model["model_name"])

        # Should select different models for different tasks
        assert len(models_selected) > 1, "Should select variety of models for different tasks"

    def test_protocol_switching(self, api_client, base_url, health_check):
        """Test that protocol switches between standard and minion based on complexity."""
        simple_payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-4",
        }
        
        complex_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Implement a distributed cache system with LRU eviction, write-through and write-behind strategies, sharding, and fault tolerance.",
                }
            ],
            "model": "gpt-4",
        }

        # Test simple request
        simple_response = api_client.post(
            f"{base_url}/predict",
            json=simple_payload,
            timeout=30,
        )
        
        # Test complex request
        complex_response = api_client.post(
            f"{base_url}/predict",
            json=complex_payload,
            timeout=30,
        )

        assert simple_response.status_code == 200
        assert complex_response.status_code == 200

        simple_protocol = simple_response.json()["output"][0]["protocol_info"]["protocol"]
        complex_protocol = complex_response.json()["output"][0]["protocol_info"]["protocol"]

        # Protocols might differ based on complexity
        # This is just to verify both protocols work
        assert simple_protocol in ["standard", "minion"]
        assert complex_protocol in ["standard", "minion"]

    def test_error_recovery(self, api_client, base_url, health_check):
        """Test service recovery from errors."""
        # Send invalid request
        invalid_payload = {"invalid": "data"}
        response1 = api_client.post(
            f"{base_url}/predict",
            json=invalid_payload,
            timeout=30,
        )
        
        # Should return error but not crash
        assert response1.status_code in [400, 422]
        
        # Send valid request after error
        valid_payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }
        response2 = api_client.post(
            f"{base_url}/predict",
            json=valid_payload,
            timeout=30,
        )
        
        # Should recover and process normally
        assert response2.status_code == 200

    def test_memory_stability(self, api_client, base_url, health_check):
        """Test memory stability under repeated requests."""
        payload = {
            "messages": [{"role": "user", "content": "Test memory"}],
            "model": "gpt-4",
        }
        
        # Make 20 requests
        for i in range(20):
            response = api_client.post(
                f"{base_url}/predict",
                json=payload,
                timeout=30,
            )
            assert response.status_code == 200
            
            # Small delay to prevent overwhelming
            if i % 5 == 0:
                time.sleep(0.1)

        # Service should still be responsive
        health_response = api_client.get(f"{base_url}/health", timeout=5)
        assert health_response.status_code == 200


class TestDomainClassification:
    """Test domain-specific classification accuracy."""

    @pytest.fixture(scope="class")
    def api_client(self):
        """Create a requests session."""
        return requests.Session()

    @pytest.fixture(scope="class")
    def base_url(self):
        """Base URL for the adaptive-ai service."""
        return "http://localhost:8000"

    def test_health_domain(self, api_client, base_url):
        """Test health domain classification."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the symptoms of diabetes and how can it be managed?",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(f"{base_url}/predict", json=payload, timeout=30)
        assert response.status_code == 200
        
        # Check if health domain is detected
        result = response.json()
        classification = result["output"][0]["classification_result"]
        # Domain classification should be present
        assert "domain" in classification or "domains" in classification

    def test_business_domain(self, api_client, base_url):
        """Test business domain classification."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a SWOT analysis for entering the electric vehicle market.",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(f"{base_url}/predict", json=payload, timeout=30)
        assert response.status_code == 200

    def test_science_domain(self, api_client, base_url):
        """Test science domain classification."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the process of photosynthesis at the molecular level.",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(f"{base_url}/predict", json=payload, timeout=30)
        assert response.status_code == 200

    def test_technology_domain(self, api_client, base_url):
        """Test technology domain classification."""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "How does blockchain technology ensure data immutability?",
                }
            ],
            "model": "gpt-4",
        }

        response = api_client.post(f"{base_url}/predict", json=payload, timeout=30)
        assert response.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])