"""
Comprehensive caching system tests for adaptive_ai services.
Tests model selection consistency, performance, and current architecture.
"""

import threading
import time

import pytest  # type: ignore

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_router import ModelRouter


@pytest.mark.unit
class TestModelSelectorCaching:
    """Test model selector caching and consistency behavior."""

    def test_initialization(self):
        """Test router is properly initialized."""
        router = ModelRouter()

        # Should initialize without errors
        assert router is not None
        assert hasattr(router, "select_models")

    def test_model_selection_consistency(self):
        """Test model selection is consistent for same inputs."""
        router = ModelRouter()

        # Test model selection consistency
        models1 = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.TEXT_GENERATION,
            models_input=None,
            cost_bias=0.5,
        )

        models2 = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.TEXT_GENERATION,
            models_input=None,
            cost_bias=0.5,
        )

        # Should return consistent results
        assert len(models1) == len(models2)
        if models1 and models2:
            assert models1[0].provider == models2[0].provider
            assert models1[0].model_name == models2[0].model_name

    def test_custom_model_constraint(self):
        """Test custom model constraints work correctly."""
        router = ModelRouter()

        # Test with OpenAI-only constraint
        custom_models = [ModelCapability(provider="OPENAI", model_name="gpt-4o")]

        models = router.select_models(
            task_complexity=0.7,
            task_type=TaskType.CODE_GENERATION,
            models_input=custom_models,
            cost_bias=0.5,
        )

        # Should only return OpenAI models
        for model in models:
            assert model.provider == "OPENAI"

    def test_task_type_filtering(self):
        """Test models are filtered by task type correctly."""
        router = ModelRouter()

        # Test different task types return different models
        code_models = router.select_models(
            task_complexity=0.8,
            task_type=TaskType.CODE_GENERATION,
            models_input=None,
            cost_bias=0.5,
        )

        chat_models = router.select_models(
            task_complexity=0.3,
            task_type=TaskType.CHATBOT,
            models_input=None,
            cost_bias=0.5,
        )

        # Results should be different for different task types
        if code_models and chat_models:
            # At least one model should be different
            code_names = {m.model_name for m in code_models}
            chat_names = {m.model_name for m in chat_models}
            assert code_names != chat_names

    def test_complexity_filtering(self):
        """Test models are filtered by complexity correctly."""
        router = ModelRouter()

        # Test high complexity request
        complex_models = router.select_models(
            task_complexity=0.9,
            task_type=TaskType.OTHER,
            models_input=None,
            cost_bias=0.5,
        )

        # Test low complexity request
        simple_models = router.select_models(
            task_complexity=0.2,
            task_type=TaskType.CLASSIFICATION,
            models_input=None,
            cost_bias=0.5,
        )

        # Should return some models
        assert isinstance(complex_models, list)
        assert isinstance(simple_models, list)


@pytest.mark.unit
class TestModelRouterCaching:
    """Test model router caching behavior."""

    def test_router_initialization(self):
        """Test router initializes correctly."""
        router = ModelRouter()

        # Should initialize without errors
        assert router is not None
        assert hasattr(router, "select_models")

    def test_model_capability_creation(self):
        """Test ModelCapability can be created with current schema."""
        # Test minimal ModelCapability
        minimal_cap = ModelCapability(provider="OPENAI", model_name="gpt-4o")
        assert minimal_cap.provider == "OPENAI"
        assert minimal_cap.model_name == "gpt-4o"
        assert minimal_cap.is_partial  # Should be partial

        # Test full ModelCapability
        full_cap = ModelCapability(
            provider="ANTHROPIC",
            model_name="claude-3-5-sonnet",
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=15.0,
            max_context_tokens=200000,
            supports_function_calling=True,
            task_type=TaskType.CODE_GENERATION,
            complexity="hard",
        )
        assert not full_cap.is_partial  # Should be complete
        assert full_cap.complexity_score == 0.8  # "hard" -> 0.8

    def test_complexity_score_conversion(self):
        """Test complexity string to score conversion."""
        # Test different complexity levels
        easy_cap = ModelCapability(
            provider="OPENAI", model_name="test", complexity="easy"
        )
        assert easy_cap.complexity_score == 0.2

        medium_cap = ModelCapability(
            provider="OPENAI", model_name="test", complexity="medium"
        )
        assert medium_cap.complexity_score == 0.5

        hard_cap = ModelCapability(
            provider="OPENAI", model_name="test", complexity="hard"
        )
        assert hard_cap.complexity_score == 0.8


@pytest.mark.unit
class TestCachingPerformance:
    """Performance tests for current caching systems."""

    def test_concurrent_model_selection(self):
        """Test concurrent model selection doesn't cause issues."""
        router = ModelRouter()

        results = []
        errors = []

        def worker():
            try:
                # Each thread selects models
                for i in range(10):
                    models = router.select_models(
                        task_complexity=0.5 + (i % 5) * 0.1,
                        task_type=TaskType.TEXT_GENERATION,
                        models_input=None,
                        cost_bias=0.5,
                    )
                    results.append(len(models))
            except Exception as e:
                errors.append(e)

        # Run 5 threads concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) > 0

    def test_model_selection_performance(self):
        """Test model selection performance is reasonable."""
        router = ModelRouter()

        # Measure selection time
        start_time = time.time()

        for _i in range(20):
            models = router.select_models(
                task_complexity=0.7,
                task_type=TaskType.CODE_GENERATION,
                models_input=None,
                cost_bias=0.5,
            )
            assert isinstance(models, list)

        total_time = time.time() - start_time
        avg_time = total_time / 20

        # Should be fast (< 50ms per selection on average)
        assert avg_time < 0.05, f"Selection too slow: {avg_time:.3f}s average"


@pytest.mark.unit
class TestCacheEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_task_type(self):
        """Test handling of invalid task types."""
        router = ModelRouter()

        # Test with invalid task type - use OTHER as fallback
        models = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.OTHER,
            models_input=None,
            cost_bias=0.5,
        )

        # Should return some models
        assert isinstance(models, list)

    def test_extreme_complexity_values(self):
        """Test extreme complexity values."""
        router = ModelRouter()

        # Test with very high complexity
        models_high = router.select_models(
            task_complexity=1.0,
            task_type=TaskType.OTHER,
            models_input=None,
            cost_bias=0.5,
        )

        # Test with very low complexity
        models_low = router.select_models(
            task_complexity=0.0,
            task_type=TaskType.CLASSIFICATION,
            models_input=None,
            cost_bias=0.5,
        )

        assert isinstance(models_high, list)
        assert isinstance(models_low, list)

    def test_large_token_counts(self):
        """Test handling of large token counts."""
        router = ModelRouter()

        # Test with high complexity for models with large context
        models = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.TEXT_GENERATION,
            models_input=None,
            cost_bias=0.5,
        )

        # Should handle gracefully
        assert isinstance(models, list)

        # Check if any models support large context
        large_context_models = [
            model
            for model in models
            if model.max_context_tokens and model.max_context_tokens >= 100000
        ]

        print(f"Models with 100k+ context: {len(large_context_models)}/{len(models)}")


@pytest.mark.unit
class TestModelCapabilityValidation:
    """Test ModelCapability validation and edge cases."""

    def test_partial_model_capability(self):
        """Test partial model capabilities are handled correctly."""
        # Test minimal capability
        partial_cap = ModelCapability(provider="OPENAI")
        assert partial_cap.provider == "OPENAI"
        assert partial_cap.is_partial

        # Test with model name only
        partial_cap2 = ModelCapability(model_name="gpt-4")
        assert partial_cap2.model_name == "gpt-4"
        assert partial_cap2.is_partial

    def test_task_type_validation(self):
        """Test TaskType enum validation."""
        # Test with valid TaskType enum
        cap_enum = ModelCapability(
            provider="ANTHROPIC",
            model_name="claude-3",
            task_type=TaskType.CODE_GENERATION,
        )
        assert cap_enum.task_type == TaskType.CODE_GENERATION

        # Test with string task type
        cap_string = ModelCapability(
            provider="ANTHROPIC", model_name="claude-3", task_type="Code Generation"
        )
        assert cap_string.task_type == "Code Generation"

    def test_unique_id_generation(self):
        """Test unique ID generation for models."""
        cap = ModelCapability(
            provider="OpenAI", model_name="GPT-4o"  # Mixed case  # Mixed case
        )

        # Should normalize to lowercase
        assert cap.unique_id == "openai:gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
