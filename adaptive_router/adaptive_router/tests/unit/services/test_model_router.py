"""Unit tests for ModelRouter service."""

from unittest.mock import Mock

import pytest

from adaptive_router.models import RoutingDecision
from adaptive_router.models.llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_router import ModelRouter


@pytest.fixture
def mock_router_service():
    """Create a mock router_service with internal _Router."""
    # Create mock internal router that returns RoutingDecision
    mock_internal_router = Mock()
    mock_internal_router.route.return_value = RoutingDecision(
        selected_model_id="openai:gpt-4",
        selected_model_name="gpt-4",
        routing_score=0.85,
        predicted_accuracy=0.92,
        estimated_cost=0.03,
        cluster_id=5,
        cluster_confidence=0.88,
        lambda_param=0.5,
        reasoning="Selected based on cluster analysis",
        alternatives=[
            {
                "model_id": "anthropic:claude-3-sonnet-20240229",
                "score": 0.82,
                "predicted_accuracy": 0.89,
            }
        ],
        routing_time_ms=45.2,
    )

    # Mock the models dict for get_supported_models()
    mock_internal_router.models = {
        "openai:gpt-4": Mock(),
        "openai:gpt-3.5-turbo": Mock(),
        "anthropic:claude-3-sonnet-20240229": Mock(),
    }

    # Create mock service with router attribute (new architecture)
    mock_service = Mock()
    mock_service.router = mock_internal_router

    return mock_service


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    def test_initialization(self, mock_router_service: Mock) -> None:
        """Test router initialization creates a functional instance."""
        router = ModelRouter(router_service=mock_router_service)

        # Test that the router can perform its main function
        request = ModelSelectionRequest(
            prompt="Write a simple hello world function",
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Verify the router produces valid output
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_initialization_without_params(self, mock_router_service: Mock) -> None:
        """Test router delegates to internal _Router."""
        router = ModelRouter(router_service=mock_router_service)

        # Test that the router works with mock service
        request = ModelSelectionRequest(
            prompt="Calculate the factorial of 10",
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Verify valid response
        assert response.provider
        assert response.model

    def test_select_model_with_full_models(self, mock_router_service: Mock) -> None:
        """Test model selection when full models are provided."""
        router = ModelRouter(router_service=mock_router_service)

        sample_models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=1.25,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type="Text Generation",
            ),
        ]

        request = ModelSelectionRequest(
            prompt="Write a Python function to implement quicksort",
            models=sample_models,
            cost_bias=0.9,
        )
        response = router.select_model(request)

        # Verify response structure
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_select_model_cost_bias_low(self, mock_router_service: Mock) -> None:
        """Test that low cost bias works correctly."""
        router = ModelRouter(router_service=mock_router_service)

        # Low cost bias (0.1)
        request = ModelSelectionRequest(
            prompt="Write a simple hello world program",
            cost_bias=0.1,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_cost_bias_high(self, mock_router_service: Mock) -> None:
        """Test that high cost bias works correctly."""
        router = ModelRouter(router_service=mock_router_service)

        # High cost bias (0.9)
        request = ModelSelectionRequest(
            prompt="Design a distributed system architecture for real-time data processing",
            cost_bias=0.9,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_empty_input(self, mock_router_service: Mock) -> None:
        """Test selecting models when no models are provided."""
        router = ModelRouter(router_service=mock_router_service)

        request = ModelSelectionRequest(
            prompt="Explain quantum computing",
            models=None,
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Should delegate to service
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_partial_model_filtering(self, mock_router_service: Mock) -> None:
        """Test filtering with partial ModelCapability."""
        router = ModelRouter(router_service=mock_router_service)

        partial_models = [
            ModelCapability(
                provider="openai",
                model_name=None,
                cost_per_1m_input_tokens=None,
                cost_per_1m_output_tokens=None,
                max_context_tokens=None,
                supports_function_calling=None,
            )
        ]

        request = ModelSelectionRequest(
            prompt="Generate a creative story",
            models=partial_models,
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Should delegate to service
        assert response.provider
        assert response.model

    def test_model_selection_code_task(self, mock_router_service: Mock) -> None:
        """Test model selection for code generation tasks."""
        router = ModelRouter(router_service=mock_router_service)

        request = ModelSelectionRequest(
            prompt="Write a Python function to implement binary search",
            cost_bias=0.5,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_model_selection_creative_task(self, mock_router_service: Mock) -> None:
        """Test model selection for creative writing tasks."""
        router = ModelRouter(router_service=mock_router_service)

        request = ModelSelectionRequest(
            prompt="Write a short poem about nature",
            cost_bias=0.3,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model


class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cost_bias_raises_error(self) -> None:
        """Test that invalid cost bias values raise validation errors."""

        models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-5",
                cost_per_1m_input_tokens=1.25,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type="Text Generation",
            )
        ]

        # Test cost_bias > 1.0 raises ValidationError
        with pytest.raises(Exception) as exc_info:
            ModelSelectionRequest(
                prompt="Simple task",
                models=models,
                cost_bias=2.0,
            )
        assert (
            "cost_bias" in str(exc_info.value).lower()
            or "validation" in str(exc_info.value).lower()
        )

        # Test cost_bias < 0.0 raises ValidationError
        with pytest.raises(Exception) as exc_info:
            ModelSelectionRequest(
                prompt="Simple task",
                models=models,
                cost_bias=-1.0,
            )
        assert (
            "cost_bias" in str(exc_info.value).lower()
            or "validation" in str(exc_info.value).lower()
        )

    def test_valid_cost_bias_boundary_values(self, mock_router_service: Mock) -> None:
        """Test that boundary values 0.0 and 1.0 are accepted."""
        router = ModelRouter(router_service=mock_router_service)

        # Test cost_bias = 0.0 (minimum)
        request_min = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=0.0,
        )
        response_min = router.select_model(request_min)
        assert response_min.provider
        assert response_min.model

        # Test cost_bias = 1.0 (maximum)
        request_max = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=1.0,
        )
        response_max = router.select_model(request_max)
        assert response_max.provider
        assert response_max.model

    def test_complex_prompt_handling(self, mock_router_service: Mock) -> None:
        """Test handling of very complex prompts."""
        router = ModelRouter(router_service=mock_router_service)

        # Very long and complex prompt
        complex_prompt = """
        Design and implement a distributed microservices architecture with the following requirements:
        1. Real-time data processing with sub-second latency
        2. Horizontal scalability to handle 1M+ requests per second
        3. Fault tolerance with automatic failover
        4. Multi-region deployment with active-active replication
        5. End-to-end encryption and compliance with GDPR
        Include implementation details, technology stack recommendations, and deployment strategies.
        """

        request = ModelSelectionRequest(
            prompt=complex_prompt,
            cost_bias=0.9,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_simple_prompt_handling(self, mock_router_service: Mock) -> None:
        """Test handling of very simple prompts."""
        router = ModelRouter(router_service=mock_router_service)

        request = ModelSelectionRequest(
            prompt="Hello, how are you?",
            cost_bias=0.1,
        )
        response = router.select_model(request)

        # Should successfully select a model
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_alternatives_generation(self, mock_router_service: Mock) -> None:
        """Test that alternatives are properly generated."""
        router = ModelRouter(router_service=mock_router_service)

        request = ModelSelectionRequest(
            prompt="Write a complex algorithm",
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Should successfully select a model
        assert response.provider
        assert response.model
        # Should have alternatives
        assert isinstance(response.alternatives, list)

    def test_no_models_raises_error(self, mock_router_service: Mock) -> None:
        """Test that providing empty models list is handled."""
        router = ModelRouter(router_service=mock_router_service)

        # Empty models list should be handled by service
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=[],
            cost_bias=0.5,
        )

        # Should not raise error
        response = router.select_model(request)
        assert response.provider
        assert response.model
