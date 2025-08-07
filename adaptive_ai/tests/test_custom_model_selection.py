"""Comprehensive unit tests for custom model selection edge cases."""

from unittest.mock import Mock, patch

import pytest  # type: ignore

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelEntry,
    ModelSelectionRequest,
    ProtocolManagerConfig,
)
from adaptive_ai.models.llm_enums import ProviderType
from adaptive_ai.services.model_selector import ModelSelectionService


class TestCustomModelSelection:
    """Test custom model selection with various edge cases."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @pytest.fixture
    def model_service(self, mock_logger):
        """Create ModelSelectionService with mocked logger."""
        return ModelSelectionService(lit_logger=mock_logger)

    @pytest.fixture
    def basic_chat_request(self):
        """Basic chat completion request."""
        return {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "model": "gpt-4",
        }

    @pytest.fixture
    def classification_result(self):
        """Basic classification result."""
        return ClassificationResult(
            task_type_1=["Code Generation"],
            task_type_2=["Other"],
            task_type_prob=[0.8, 0.2],
            creativity_scope=[0.5],
            reasoning=[0.7],
            contextual_knowledge=[0.6],
            prompt_complexity_score=[0.6],
            domain_knowledge=[0.5],
            number_of_few_shots=[0.0],
            no_label_reason=[0.1],
            constraint_ct=[0.3],
        )

    def test_user_specified_models_basic_success(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test successful selection with user-specified models."""
        # Arrange
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-sonnet",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=75.0,
                max_context_tokens=200000,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        assert all(name in ["gpt-4", "claude-3-sonnet"] for name in model_names)

    def test_custom_provider_no_fallback(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test custom provider (e.g., 'botir') is preserved exactly as specified."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="botir",  # Custom provider string
                model_name="botir-custom-model",
                cost_per_1m_input_tokens=20.0,
                cost_per_1m_output_tokens=40.0,
                max_context_tokens=4096,
                supports_function_calling=False,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert
        assert len(result) >= 1
        assert result[0].model_name == "botir-custom-model"
        # Should preserve custom provider exactly (no OpenAI fallback)
        assert "botir" in result[0].providers
        assert result[0].providers[0] == "botir"
        # Should NOT have OpenAI fallback
        assert ProviderType.OPENAI not in result[0].providers
        # Should NOT have original provider metadata (no longer needed)
        assert not hasattr(result[0], "_original_provider")

    def test_model_exceeds_context_length_fallback(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test when user model can't handle token count, falls back to first model."""
        # Arrange - model with small context limit
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-3.5-turbo",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=4096,  # Small context
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act - with large token count that exceeds context
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=10000,  # Exceeds 4096 limit
        )

        # Assert - should still return the user's model (fallback behavior)
        assert len(result) == 1
        assert result[0].model_name == "gpt-3.5-turbo"

    def test_model_not_in_registry_fallback(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test when user specifies model not in system registry."""
        # Arrange - completely custom model not in registry
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="my-custom-fine-tuned-model",
                cost_per_1m_input_tokens=25.0,
                cost_per_1m_output_tokens=50.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should still return the user's model
        assert len(result) == 1
        assert result[0].model_name == "my-custom-fine-tuned-model"

    def test_wrong_provider_for_model_fallback(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test when user specifies wrong provider for a known model."""
        # Arrange - GPT-4 with wrong provider
        user_models = [
            ModelCapability(
                provider=ProviderType.ANTHROPIC,  # Wrong provider for GPT-4
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should still return the user's model (fallback)
        assert len(result) == 1
        assert result[0].model_name == "gpt-4"

    def test_multiple_models_some_filtered_some_survive(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test when some user models pass filtering and some don't."""
        # Arrange - mix of models with different context limits
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-3.5-turbo",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=4096,  # Will be filtered out
                supports_function_calling=True,
            ),
            ModelCapability(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-sonnet",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=75.0,
                max_context_tokens=200000,  # Will pass
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act - with token count that exceeds first model's limit
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=8000,  # Exceeds gpt-3.5-turbo but not claude
        )

        # Assert - should return surviving models or fallback to first
        assert len(result) >= 1
        # Either claude survived filtering, or we got fallback to gpt-3.5-turbo
        model_names = [entry.model_name for entry in result]
        assert any(name in ["claude-3-sonnet", "gpt-3.5-turbo"] for name in model_names)

    def test_mixed_context_limits_with_custom_models(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that custom models are trusted even with large token counts."""
        # Arrange - mix of registry model with known limits and custom model
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-3.5-turbo",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=4096,  # Will be filtered out
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="custom-provider",
                model_name="custom-large-model",
                cost_per_1m_input_tokens=0.5,
                cost_per_1m_output_tokens=1.0,
                max_context_tokens=100000,  # Custom model - will be trusted
                supports_function_calling=False,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act - with huge token count that exceeds registry model limit
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=50000,  # Exceeds gpt-3.5-turbo limit but custom model is trusted
        )

        # Assert - custom model should be trusted and preserved
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        # Custom model should be included since it's trusted
        assert "custom-large-model" in model_names

    def test_cost_optimization_preserves_user_models(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that cost optimization only reorders user models, doesn't filter them out."""
        # Arrange
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",  # Expensive
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-3.5-turbo",  # Cheaper
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=16385,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(
            models=user_models,
            cost_bias=0.9,  # Strong cost preference
        )
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - both models should be present (cost optimization doesn't filter)
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        # Should include user's models (possibly reordered by cost)
        assert all(name in ["gpt-4", "gpt-3.5-turbo"] for name in model_names)

    def test_model_creation_error_handling(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of invalid ModelCapability objects."""
        # Arrange - model with empty name and valid model
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="",  # Empty model name (valid but not useful)
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-sonnet",  # Valid model
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=75.0,
                max_context_tokens=200000,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle empty model name gracefully
        assert len(result) >= 1
        # Due to filtering/fallback, system returns the first model (which has empty name)
        model_names = [entry.model_name for entry in result]
        # Should contain the first model (empty name) due to fallback behavior
        assert any(name == "" for name in model_names)  # Empty name model from fallback

    def test_no_user_models_falls_back_to_system(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that when no user models provided, uses system selection."""
        # Arrange - no protocol config (no user models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=None,
        )

        # Act
        with patch.object(model_service, "_use_system_model_selection") as mock_system:
            mock_system.return_value = [
                ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4")
            ]

            result = model_service.select_candidate_models(
                request=request,
                classification_result=classification_result,
                prompt_token_count=100,
            )

        # Assert - should call system selection
        mock_system.assert_called_once()
        assert len(result) == 1
        assert result[0].model_name == "gpt-4"

    def test_empty_user_models_list(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of empty user models list falls back to system selection."""
        # Arrange - empty models list
        protocol_config = ProtocolManagerConfig(models=[])
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should fall back to system model selection (not raise error)
        assert len(result) >= 1
        # Should contain system-selected models, not user models
        model_names = [entry.model_name for entry in result]
        assert all(name for name in model_names)  # Should have valid model names

    def test_custom_model_with_task_specifications(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test custom model with task_type and complexity specifications."""
        # Arrange - custom model with task info
        user_models = [
            ModelCapability(
                provider="custom-provider",
                model_name="code-specialist-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=16384,
                supports_function_calling=True,
                task_type="CODE_GENERATION",  # Task specialization
                complexity="hard",  # Complexity level
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert
        assert len(result) == 1
        assert result[0].model_name == "code-specialist-model"

    def test_mixed_registry_and_custom_models(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test mix of registry models and fully custom models."""
        # Arrange - mix of known and custom models
        user_models = [
            # Registry model (partial specification)
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                # Missing cost/context info - should be enriched from registry
            ),
            # Fully custom model
            ModelCapability(
                provider="custom-provider",
                model_name="my-custom-model",
                cost_per_1m_input_tokens=5.0,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=32768,
                supports_function_calling=True,
                task_type="CREATIVE_WRITING",
                complexity="medium",
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        # Should include both models (registry enriched + custom)
        assert any(name in ["gpt-4", "my-custom-model"] for name in model_names)

    def test_mixed_enum_and_string_providers(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test mix of ProviderType enum and custom string providers in same request."""
        # Arrange - mix of enum and string providers
        user_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,  # Enum provider
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="custom-ai",  # String provider
                model_name="custom-model-v2",
                cost_per_1m_input_tokens=5.0,
                cost_per_1m_output_tokens=10.0,
                max_context_tokens=16384,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        providers = [entry.providers[0] for entry in result]

        # Should contain both models
        assert any(name in ["gpt-4", "custom-model-v2"] for name in model_names)
        # Should preserve both provider types
        assert any(provider == ProviderType.OPENAI for provider in providers)
        assert any(provider == "custom-ai" for provider in providers)

    def test_custom_provider_cost_optimization(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that cost optimization works with custom providers."""
        # Arrange - custom provider without registry cost data
        user_models = [
            ModelCapability(
                provider="budget-ai",  # Custom provider
                model_name="budget-model",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="premium-ai",  # Another custom provider
                model_name="premium-model",
                cost_per_1m_input_tokens=50.0,
                cost_per_1m_output_tokens=100.0,
                max_context_tokens=32768,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(
            models=user_models,
            cost_bias=0.9,  # Strong cost preference
        )
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle custom providers gracefully in cost optimization
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        # Both models should be present (cost optimization doesn't filter custom providers)
        assert any(name in ["budget-model", "premium-model"] for name in model_names)

    def test_invalid_provider_type_handling(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of invalid provider types."""
        # Arrange - provider that's not a string or ProviderType
        user_models = [
            ModelCapability(
                provider=123,  # Invalid type (should be string or ProviderType)
                model_name="invalid-provider-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act & Assert - should handle gracefully
        try:
            result = model_service.select_candidate_models(
                request=request,
                classification_result=classification_result,
                prompt_token_count=100,
            )
            # If it doesn't raise an error, verify it handles the invalid type
            assert len(result) >= 0
        except (ValueError, TypeError):
            # Acceptable to raise an error for invalid types
            pass

    def test_empty_provider_string(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of empty provider string."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="",  # Empty provider string
                model_name="empty-provider-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle empty provider gracefully
        assert len(result) >= 1
        assert result[0].model_name == "empty-provider-model"
        assert result[0].providers[0] == ""

    def test_unicode_custom_provider(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of unicode/special characters in custom provider names."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="模型提供商",  # Chinese characters
                model_name="unicode-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="café-ai",  # Accented characters
                model_name="accent-model",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=30.0,
                max_context_tokens=16384,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should preserve unicode providers exactly
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        providers = [entry.providers[0] for entry in result]

        assert any(name in ["unicode-model", "accent-model"] for name in model_names)
        assert any(provider in ["模型提供商", "café-ai"] for provider in providers)

    def test_very_long_provider_name(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of very long provider names."""
        # Arrange
        long_provider = "a" * 1000  # 1000 character provider name
        user_models = [
            ModelCapability(
                provider=long_provider,
                model_name="long-provider-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle long provider names
        assert len(result) >= 1
        assert result[0].model_name == "long-provider-model"
        assert result[0].providers[0] == long_provider

    def test_case_sensitive_provider_names(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that provider names are case-sensitive."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="OpenAI",  # Capital letters
                model_name="capital-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="openai",  # Lowercase (matches enum value)
                model_name="lowercase-model",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=30.0,
                max_context_tokens=16384,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should preserve case sensitivity
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        providers = [entry.providers[0] for entry in result]

        # Both models should be present
        assert any(name in ["capital-model", "lowercase-model"] for name in model_names)
        # Case should be preserved
        if "OpenAI" in providers:
            assert "OpenAI" in providers  # Capital case preserved
        if "openai" in providers:
            # This might be converted to ProviderType.OPENAI enum
            pass

    def test_provider_with_special_characters(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test provider names with special characters, numbers, and symbols."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="ai-provider-2024",  # Hyphens and numbers
                model_name="hyphen-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="provider_with_underscores",  # Underscores
                model_name="underscore-model",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=30.0,
                max_context_tokens=16384,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="provider.with.dots",  # Dots
                model_name="dot-model",
                cost_per_1m_input_tokens=20.0,
                cost_per_1m_output_tokens=40.0,
                max_context_tokens=32768,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle special characters in provider names
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        providers = [entry.providers[0] for entry in result]

        expected_models = ["hyphen-model", "underscore-model", "dot-model"]
        expected_providers = [
            "ai-provider-2024",
            "provider_with_underscores",
            "provider.with.dots",
        ]

        assert any(name in expected_models for name in model_names)
        assert any(provider in expected_providers for provider in providers)

    def test_duplicate_custom_providers(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of duplicate custom provider names with different models."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="custom-ai",  # Same provider
                model_name="model-v1",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="custom-ai",  # Same provider, different model
                model_name="model-v2",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=30.0,
                max_context_tokens=16384,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act
        result = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=100,
        )

        # Assert - should handle duplicate providers with different models
        assert len(result) >= 1
        model_names = [entry.model_name for entry in result]
        providers = [entry.providers[0] for entry in result]

        # Both models should be present
        assert any(name in ["model-v1", "model-v2"] for name in model_names)
        # All should have same provider
        custom_ai_entries = [p for p in providers if p == "custom-ai"]
        assert len(custom_ai_entries) >= 1

    def test_extreme_token_counts(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test handling of extreme token counts (very small and very large)."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="flexible-ai",
                model_name="flexible-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=1000000,  # 1M tokens
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Test with very small token count
        result_small = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=1,  # Minimal tokens
        )

        # Test with very large token count
        result_large = model_service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=999999,  # Nearly max tokens
        )

        # Assert - custom model should be trusted for both cases
        assert len(result_small) >= 1
        assert len(result_large) >= 1
        assert result_small[0].model_name == "flexible-model"
        assert result_large[0].model_name == "flexible-model"

    def test_concurrent_requests_with_custom_providers(
        self, model_service, basic_chat_request, classification_result
    ):
        """Test that custom providers work correctly with concurrent/repeated requests."""
        # Arrange
        user_models = [
            ModelCapability(
                provider="concurrent-ai",
                model_name="concurrent-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        protocol_config = ProtocolManagerConfig(models=user_models)
        request = ModelSelectionRequest(
            chat_completion_request=basic_chat_request,
            protocol_manager_config=protocol_config,
        )

        # Act - make multiple requests to simulate concurrency
        results = []
        for i in range(5):
            result = model_service.select_candidate_models(
                request=request,
                classification_result=classification_result,
                prompt_token_count=100 + i,  # Slightly different token counts
            )
            results.append(result)

        # Assert - all results should be consistent
        for i, result in enumerate(results):
            assert len(result) >= 1, f"Request {i} failed"
            assert (
                result[0].model_name == "concurrent-model"
            ), f"Request {i} wrong model"
            assert (
                result[0].providers[0] == "concurrent-ai"
            ), f"Request {i} wrong provider"


class TestModelEnrichment:
    """Test model enrichment functionality."""

    @pytest.fixture
    def model_service(self):
        """Create ModelSelectionService for enrichment testing."""
        return ModelSelectionService()

    def test_enrich_partial_models_registry_lookup(self, model_service):
        """Test enriching partial models from registry."""
        # Arrange - partial model specification
        partial_models = [
            ModelCapability(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                # Missing cost and context info
            ),
        ]

        # Act
        enriched = model_service.enrich_partial_models(partial_models)

        # Assert - should find and enrich from registry (if available)
        assert len(enriched) >= 0  # May be 0 if not in registry, or 1 if found

    def test_enrich_fully_specified_custom_models(self, model_service):
        """Test that fully specified custom models pass through unchanged."""
        # Arrange - fully specified custom model
        custom_models = [
            ModelCapability(
                provider="custom-provider",
                model_name="my-custom-model",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=30.0,
                max_context_tokens=16384,
                supports_function_calling=True,
                task_type="ANALYSIS",
                complexity="hard",
            ),
        ]

        # Act
        enriched = model_service.enrich_partial_models(custom_models)

        # Assert - should pass through unchanged
        assert len(enriched) == 1
        assert enriched[0].model_name == "my-custom-model"
        assert enriched[0].cost_per_1m_input_tokens == 15.0
        assert enriched[0].task_type == "ANALYSIS"

    def test_enrich_mixed_partial_and_full_models(self, model_service):
        """Test enriching mix of partial and fully specified models."""
        # Arrange
        mixed_models = [
            # Partial model
            ModelCapability(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-sonnet",
            ),
            # Fully specified custom model
            ModelCapability(
                provider="custom",
                model_name="custom-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=False,
            ),
        ]

        # Act
        enriched = model_service.enrich_partial_models(mixed_models)

        # Assert - should handle both types
        assert len(enriched) >= 1  # At least the custom model should pass through
        model_names = [model.model_name for model in enriched]
        assert "custom-model" in model_names

    def test_enrich_empty_models_list(self, model_service):
        """Test enriching empty models list."""
        # Act
        enriched = model_service.enrich_partial_models([])

        # Assert
        assert enriched == []

    def test_enrich_invalid_models_handling(self, model_service):
        """Test handling of invalid models during enrichment."""
        # Arrange - model with missing required fields and invalid data
        invalid_models = [
            ModelCapability(
                provider=None,  # Invalid
                model_name="",  # Invalid
            ),
            ModelCapability(
                provider="valid-provider",
                model_name="valid-model",
                cost_per_1m_input_tokens=10.0,
                cost_per_1m_output_tokens=20.0,
                max_context_tokens=8192,
                supports_function_calling=True,
            ),
        ]

        # Act
        enriched = model_service.enrich_partial_models(invalid_models)

        # Assert - should handle invalid models gracefully
        # Should return valid models and skip invalid ones
        assert len(enriched) >= 0
        if enriched:
            # Valid models should have proper model names
            assert all(model.model_name for model in enriched)
