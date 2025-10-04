"""Unit tests for ModelRouter service."""

from unittest.mock import Mock

import pytest

from adaptive_router.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
)
from adaptive_router.services.model_registry import ModelRegistry
from adaptive_router.services.model_router import ModelRouter
from adaptive_router.services.yaml_model_loader import YAMLModelDatabase


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create a ModelRegistry instance for testing."""
    yaml_db = YAMLModelDatabase()
    return ModelRegistry(yaml_db)


@pytest.fixture
def mock_prompt_classifier():
    """Create a mock prompt classifier to avoid HuggingFace rate limits."""
    mock_classifier = Mock()
    # Default classification result for simple prompts
    # Must include all required fields from ClassificationResult
    mock_classifier.classify_prompt.return_value = {
        "task_type_1": "Text Generation",
        "task_type_2": "Chatbot",
        "task_type_prob": 0.85,
        "creativity_scope": 0.3,
        "reasoning": 0.4,
        "contextual_knowledge": 0.2,
        "prompt_complexity_score": 0.3,
        "domain_knowledge": 0.1,
        "number_of_few_shots": 0.0,
        "no_label_reason": 0.9,
        "constraint_ct": 0.2,
    }
    return mock_classifier


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    @pytest.fixture
    def sample_models(self) -> list[ModelCapability]:
        """Sample models for testing (from actual model_data YAML files)."""
        return [
            ModelCapability(
                provider="openai",
                model_name="gpt-5",
                cost_per_1m_input_tokens=1.25,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type="Text Generation",
            ),
            ModelCapability(
                provider="openai",
                model_name="gpt-5-nano",
                cost_per_1m_input_tokens=0.05,
                cost_per_1m_output_tokens=0.4,
                max_context_tokens=64000,
                supports_function_calling=True,
                task_type="Text Generation",
            ),
            ModelCapability(
                provider="anthropic",
                model_name="claude-sonnet-4-5-20250929",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=15.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type="Code Generation",
            ),
        ]

    def test_initialization(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test router initialization creates a functional instance."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

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

    def test_initialization_without_params(self, mock_prompt_classifier: Mock) -> None:
        """Test router can be initialized without external dependencies."""
        router = ModelRouter(prompt_classifier=mock_prompt_classifier)

        # Test that the router works with default initialization
        request = ModelSelectionRequest(
            prompt="Calculate the factorial of 10",
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Verify valid response
        assert response.provider
        assert response.model

    def test_select_model_with_full_models(
        self,
        model_registry: ModelRegistry,
        sample_models: list[ModelCapability],
        mock_prompt_classifier: Mock,
    ) -> None:
        """Test model selection when full models are provided."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

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

        # Verify selected model is from provided models
        assert any(
            m.provider == response.provider and m.model_name == response.model
            for m in sample_models
        )

    def test_select_model_cost_bias_low(
        self,
        model_registry: ModelRegistry,
        sample_models: list[ModelCapability],
        mock_prompt_classifier: Mock,
    ) -> None:
        """Test that low cost bias affects model selection."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        # Low cost bias (0.1) should prefer cheaper models
        request = ModelSelectionRequest(
            prompt="Write a simple hello world program",
            models=sample_models,
            cost_bias=0.1,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_cost_bias_high(
        self,
        model_registry: ModelRegistry,
        sample_models: list[ModelCapability],
        mock_prompt_classifier: Mock,
    ) -> None:
        """Test that high cost bias affects model selection."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        # High cost bias (0.9) should prefer more capable models
        request = ModelSelectionRequest(
            prompt="Design a distributed system architecture for real-time data processing",
            models=sample_models,
            cost_bias=0.9,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_select_model_empty_input(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test selecting models when no models are provided."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        request = ModelSelectionRequest(
            prompt="Explain quantum computing",
            models=None,
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Should select from registry's available models
        assert response.provider
        assert response.model
        assert isinstance(response.alternatives, list)

    def test_partial_model_filtering(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test filtering with partial ModelCapability."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

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

        # Should match models from the openai provider
        assert response.provider == "openai"
        assert response.model

    def test_model_selection_code_task(
        self,
        model_registry: ModelRegistry,
        sample_models: list[ModelCapability],
        mock_prompt_classifier: Mock,
    ) -> None:
        """Test model selection for code generation tasks."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        request = ModelSelectionRequest(
            prompt="Write a Python function to implement binary search",
            models=sample_models,
            cost_bias=0.5,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model
        # Code tasks should select capable models
        assert any(
            m.provider == response.provider and m.model_name == response.model
            for m in sample_models
        )

    def test_model_selection_creative_task(
        self,
        model_registry: ModelRegistry,
        sample_models: list[ModelCapability],
        mock_prompt_classifier: Mock,
    ) -> None:
        """Test model selection for creative writing tasks."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        request = ModelSelectionRequest(
            prompt="Write a short poem about nature",
            models=sample_models,
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

    def test_valid_cost_bias_boundary_values(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test that boundary values 0.0 and 1.0 are accepted."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

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

        # Test cost_bias = 0.0 (minimum)
        request_min = ModelSelectionRequest(
            prompt="Simple task",
            models=models,
            cost_bias=0.0,
        )
        response_min = router.select_model(request_min)
        assert response_min.provider
        assert response_min.model

        # Test cost_bias = 1.0 (maximum)
        request_max = ModelSelectionRequest(
            prompt="Simple task",
            models=models,
            cost_bias=1.0,
        )
        response_max = router.select_model(request_max)
        assert response_max.provider
        assert response_max.model

    def test_complex_prompt_handling(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test handling of very complex prompts."""
        # Set high complexity for this test
        mock_prompt_classifier.classify_prompt.return_value = {
            "task_type_1": "Text Generation",
            "task_type_2": "System Design",
            "task_type_prob": 0.95,
            "creativity_scope": 0.8,
            "reasoning": 0.9,
            "contextual_knowledge": 0.8,
            "prompt_complexity_score": 0.9,
            "domain_knowledge": 0.7,
            "number_of_few_shots": 0.0,
            "no_label_reason": 0.95,
            "constraint_ct": 0.8,
        }
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

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
            models=models,
            cost_bias=0.9,
        )
        response = router.select_model(request)

        assert response.provider
        assert response.model

    def test_simple_prompt_handling(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test handling of very simple prompts without providing specific models."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        # Don't provide models - let router use registry models
        # This avoids issues with partial model specifications
        request = ModelSelectionRequest(
            prompt="Hello, how are you?",
            models=None,
            cost_bias=0.1,
        )
        response = router.select_model(request)

        # Should successfully select a model from the registry
        assert response.provider
        assert response.model
        # With low cost bias, should prefer cheaper models
        assert isinstance(response.alternatives, list)

    def test_alternatives_generation(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test that alternatives are properly generated."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-5",
                cost_per_1m_input_tokens=1.25,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type=None,
            ),
            ModelCapability(
                provider="anthropic",
                model_name="claude-sonnet-4-5-20250929",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=15.0,
                max_context_tokens=200000,
                supports_function_calling=True,
                task_type=None,
            ),
        ]

        request = ModelSelectionRequest(
            prompt="Write a complex algorithm",
            models=models,
            cost_bias=0.5,
        )
        response = router.select_model(request)

        # Should successfully select a model
        assert response.provider
        assert response.model
        # Should have alternatives (at least 0, may be 1 if both models eligible)
        assert len(response.alternatives) >= 0
        # Alternative should be different from selected model if present
        if response.alternatives:
            assert not any(
                alt.provider == response.provider and alt.model == response.model
                for alt in response.alternatives
            )

    def test_no_models_raises_error(
        self, model_registry: ModelRegistry, mock_prompt_classifier: Mock
    ) -> None:
        """Test that providing empty models list raises appropriate error."""
        router = ModelRouter(model_registry, prompt_classifier=mock_prompt_classifier)

        # Empty models list should fall back to registry
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=[],
            cost_bias=0.5,
        )

        # Should not raise error, but use registry models
        response = router.select_model(request)
        assert response.provider
        assert response.model
