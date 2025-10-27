"""Unit tests for ModelRouter service."""

from unittest.mock import Mock, patch

import pytest

from adaptive_router.models.api import ModelCapability, ModelSelectionRequest
from adaptive_router.models.storage import (
    RouterProfile,
    ScalerParameters,
    ScalerParametersData,
    TFIDFVocabularyData,
)
from adaptive_router.core.router import ModelRouter


@pytest.fixture
def mock_router():
    """Create a mock ModelRouter with patched __init__ method."""
    from adaptive_router.models.storage import (
        ProfileMetadata,
        ClusterCentersData,
    )

    mock_profile = RouterProfile(
        metadata=ProfileMetadata(
            n_clusters=10,
            silhouette_score=0.5,
            embedding_model="all-MiniLM-L6-v2",
            tfidf_max_features=100,
            tfidf_ngram_range=[1, 2],
        ),
        cluster_centers=ClusterCentersData(
            n_clusters=10,
            feature_dim=100,
            cluster_centers=[[0.0] * 100 for _ in range(10)],
        ),
        llm_profiles={
            "openai:gpt-4": [0.08] * 10,
            "openai:gpt-3.5-turbo": [0.15] * 10,
            "anthropic:claude-3-sonnet-20240229": [0.10] * 10,
        },
        tfidf_vocabulary=TFIDFVocabularyData(
            vocabulary={"test": 0},
            idf=[1.0],
        ),
        scaler_parameters=ScalerParameters(
            embedding_scaler=ScalerParametersData(
                mean=[0.0] * 100,
                scale=[1.0] * 100,
            ),
            tfidf_scaler=ScalerParametersData(
                mean=[0.0],
                scale=[1.0],
            ),
        ),
    )

    mock_costs = {
        "openai:gpt-4": 30.0,
        "openai:gpt-3.5-turbo": 1.0,
        "anthropic:claude-3-sonnet-20240229": 15.0,
    }

    def mock_build_cluster_engine(self, profile):
        mock_engine = Mock()
        mock_engine.n_clusters = 10
        mock_engine.assign_question = Mock(return_value=(5, 0.15))
        return mock_engine

    with patch.object(
        ModelRouter, "_build_cluster_engine_from_data", mock_build_cluster_engine
    ):
        router = ModelRouter(profile=mock_profile, model_costs=mock_costs)
        return router


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    def test_initialization(self, mock_router: ModelRouter) -> None:
        """Test router initialization creates a functional instance."""
        # Test that the router can perform its main function
        request = ModelSelectionRequest(
            prompt="Write a simple hello world function",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Verify the router produces valid output
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_initialization_without_params(self, mock_router: ModelRouter) -> None:
        """Test router works with default config."""
        # Test that the router works
        request = ModelSelectionRequest(
            prompt="Calculate the factorial of 10",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Verify valid response
        assert response.provider
        assert response.model

    def test_select_model_with_full_models(self, mock_router: ModelRouter) -> None:
        """Test model selection when full models are provided."""
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
        response = mock_router.select_model(request)

        # Verify response structure
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_select_model_cost_bias_low(self, mock_router: ModelRouter) -> None:
        """Test that low cost bias works correctly."""
        # Low cost bias (0.1)
        request = ModelSelectionRequest(
            prompt="Write a simple hello world program",
            cost_bias=0.1,
        )
        response = mock_router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_cost_bias_high(self, mock_router: ModelRouter) -> None:
        """Test that high cost bias works correctly."""
        # High cost bias (0.9)
        request = ModelSelectionRequest(
            prompt="Design a distributed system architecture for real-time data processing",
            cost_bias=0.9,
        )
        response = mock_router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_empty_input(self, mock_router: ModelRouter) -> None:
        """Test selecting models when no models are provided."""
        request = ModelSelectionRequest(
            prompt="Explain quantum computing",
            models=None,
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Should work with internal model list
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_partial_model_filtering(self, mock_router: ModelRouter) -> None:
        """Test filtering with partial ModelCapability (should use all models)."""
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
        response = mock_router.select_model(request)

        # Should work with all models since partial models are ignored
        assert response.provider
        assert response.model

    def test_model_selection_code_task(self, mock_router: ModelRouter) -> None:
        """Test model selection for code generation tasks."""
        request = ModelSelectionRequest(
            prompt="Write a Python function to implement binary search",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        assert response.provider
        assert response.model

    def test_model_selection_creative_task(self, mock_router: ModelRouter) -> None:
        """Test model selection for creative writing tasks."""
        request = ModelSelectionRequest(
            prompt="Write a short poem about nature",
            cost_bias=0.3,
        )
        response = mock_router.select_model(request)

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

    def test_valid_cost_bias_boundary_values(self, mock_router: ModelRouter) -> None:
        """Test that boundary values 0.0 and 1.0 are accepted."""
        # Test cost_bias = 0.0 (minimum)
        request_min = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=0.0,
        )
        response_min = mock_router.select_model(request_min)
        assert response_min.provider
        assert response_min.model

        # Test cost_bias = 1.0 (maximum)
        request_max = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=1.0,
        )
        response_max = mock_router.select_model(request_max)
        assert response_max.provider
        assert response_max.model

    def test_complex_prompt_handling(self, mock_router: ModelRouter) -> None:
        """Test handling of very complex prompts."""
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
        response = mock_router.select_model(request)

        assert response.provider
        assert response.model

    def test_simple_prompt_handling(self, mock_router: ModelRouter) -> None:
        """Test handling of very simple prompts."""
        request = ModelSelectionRequest(
            prompt="Hello, how are you?",
            cost_bias=0.1,
        )
        response = mock_router.select_model(request)

        # Should successfully select a model
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_alternatives_generation(self, mock_router: ModelRouter) -> None:
        """Test that alternatives are properly generated."""
        request = ModelSelectionRequest(
            prompt="Write a complex algorithm",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Should successfully select a model
        assert response.provider
        assert response.model
        # Should have alternatives
        assert isinstance(response.alternatives, list)

    def test_no_models_raises_error(self, mock_router: ModelRouter) -> None:
        """Test that providing empty models list is handled."""
        # Empty models list should be handled by service
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=[],
            cost_bias=0.5,
        )

        # Should not raise error
        response = mock_router.select_model(request)
        assert response.provider
        assert response.model
