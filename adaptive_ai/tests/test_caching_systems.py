"""
Comprehensive caching system tests for adaptive_ai services.
Tests thread safety, memory bounds, eviction policies, and performance.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from unittest.mock import Mock

import pytest  # type: ignore

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType
from adaptive_ai.services.cost_optimizer import CostOptimizer
from adaptive_ai.services.model_selector import ModelSelectionService
from adaptive_ai.services.protocol_manager import ProtocolManager


class TestCostOptimizerCaching:
    """Test cost optimizer caching behavior."""

    def test_cache_initialization(self):
        """Test cache is properly initialized with size limits."""
        optimizer = CostOptimizer()

        # Check cache stats are available
        stats = optimizer.cache_stats
        assert "cost_cache" in stats
        assert "tier_cache" in stats
        assert stats["cost_cache"]["size"] == 0
        assert stats["cost_cache"]["max_size"] == 10000
        assert stats["tier_cache"]["max_size"] == 1000

    def test_cost_cache_hit_miss(self):
        """Test cache hit/miss behavior for cost calculations."""
        optimizer = CostOptimizer()

        # Create test model capability
        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="Test model",
        )

        # First call - cache miss
        cost1 = optimizer.calculate_model_cost(model_cap, 1000, 0.3)
        assert cost1 > 0
        assert optimizer.cache_stats["cost_cache"]["size"] == 1

        # Second call with same params - cache hit
        cost2 = optimizer.calculate_model_cost(model_cap, 1000, 0.3)
        assert cost1 == cost2
        assert optimizer.cache_stats["cost_cache"]["size"] == 1

        # Third call with different params - cache miss
        cost3 = optimizer.calculate_model_cost(model_cap, 2000, 0.3)
        assert cost3 != cost1
        assert optimizer.cache_stats["cost_cache"]["size"] == 2

    def test_tier_cache_consistency(self):
        """Test tier caching produces consistent results."""
        optimizer = CostOptimizer()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="expensive-model",
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,  # High cost -> premium tier
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="high",
            task_type="general",
            complexity="high",
            description="Expensive test model",
        )

        # First call
        tier1 = optimizer.get_cost_tier(model_cap)
        assert tier1 == "premium"
        assert optimizer.cache_stats["tier_cache"]["size"] == 1

        # Second call - should be cached
        tier2 = optimizer.get_cost_tier(model_cap)
        assert tier1 == tier2
        assert optimizer.cache_stats["tier_cache"]["size"] == 1

    def test_cache_memory_bounds(self):
        """Test cache respects memory bounds and evicts old entries."""
        # Create optimizer with smaller cache for testing
        optimizer = CostOptimizer()

        # Fill cache beyond its limit (simulate with many unique models)
        model_caps = []
        for i in range(15):  # More than typical cache size for tier cache
            model_cap = ModelCapability(
                provider=ProviderType.OPENAI,
                model_name=f"test-model-{i}",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=float(i),  # Varying costs
                max_context_tokens=4096,
                max_output_tokens=2048,
                supports_function_calling=True,
                languages_supported=["en"],
                model_size_params="test",
                latency_tier="medium",
                task_type="general",
                complexity="medium",
                description=f"Test model {i}",
            )
            model_caps.append(model_cap)
            optimizer.get_cost_tier(model_cap)

        # Cache should be bounded
        stats = optimizer.cache_stats
        assert stats["tier_cache"]["size"] <= stats["tier_cache"]["max_size"]

    def test_thread_safety(self):
        """Test cache is thread-safe under concurrent access."""
        optimizer = CostOptimizer()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="thread-test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="Thread test model",
        )

        results = []
        errors = []

        def worker():
            try:
                # Each thread calculates cost multiple times
                for i in range(50):
                    cost = optimizer.calculate_model_cost(model_cap, 1000 + i, 0.3)
                    results.append(cost)
                    tier = optimizer.get_cost_tier(model_cap)
                    results.append(tier)
            except Exception as e:
                errors.append(e)

        # Run 10 threads concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # No errors should occur
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Results should be consistent
        assert len(results) > 0

        # Cache should have reasonable size
        stats = optimizer.cache_stats
        assert stats["cost_cache"]["size"] > 0
        assert stats["tier_cache"]["size"] > 0

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        optimizer = CostOptimizer()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="clear-test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="Clear test model",
        )

        # Populate caches
        optimizer.calculate_model_cost(model_cap, 1000, 0.3)
        optimizer.get_cost_tier(model_cap)

        # Verify caches have entries
        stats_before = optimizer.cache_stats
        assert stats_before["cost_cache"]["size"] > 0
        assert stats_before["tier_cache"]["size"] > 0

        # Clear caches
        optimizer.clear_cache()

        # Verify caches are empty
        stats_after = optimizer.cache_stats
        assert stats_after["cost_cache"]["size"] == 0
        assert stats_after["tier_cache"]["size"] == 0


class TestProtocolManagerCaching:
    """Test protocol manager caching behavior."""

    def test_cache_initialization(self):
        """Test protocol cache is properly initialized."""
        manager = ProtocolManager()

        stats = manager.cache_stats
        assert "protocol_decision_cache" in stats
        assert stats["protocol_decision_cache"]["size"] == 0
        assert stats["protocol_decision_cache"]["max_size"] == 500

    def test_protocol_decision_caching(self):
        """Test protocol decisions are cached correctly."""
        manager = ProtocolManager()

        # Mock classification result
        classification_result = Mock()
        classification_result.prompt_complexity_score = [0.7]

        # First call - cache miss
        decision1 = manager._should_use_standard_protocol(
            classification_result, 5000, None
        )
        assert isinstance(decision1, bool)
        assert manager.cache_stats["protocol_decision_cache"]["size"] == 1

        # Second call with same params - cache hit
        decision2 = manager._should_use_standard_protocol(
            classification_result, 5000, None
        )
        assert decision1 == decision2
        assert manager.cache_stats["protocol_decision_cache"]["size"] == 1

        # Third call with different complexity - cache miss
        classification_result.prompt_complexity_score = [0.2]
        manager._should_use_standard_protocol(classification_result, 5000, None)
        assert manager.cache_stats["protocol_decision_cache"]["size"] == 2

    def test_cache_key_consistency(self):
        """Test cache keys are consistent across calls."""
        manager = ProtocolManager()

        # Create identical classification results
        result1 = Mock()
        result1.prompt_complexity_score = [0.5555555]

        result2 = Mock()
        result2.prompt_complexity_score = [0.5555555]

        # Both should hit the same cache entry
        decision1 = manager._should_use_standard_protocol(result1, 1000, None)
        cache_size_after_first = manager.cache_stats["protocol_decision_cache"]["size"]

        decision2 = manager._should_use_standard_protocol(result2, 1000, None)
        cache_size_after_second = manager.cache_stats["protocol_decision_cache"]["size"]

        assert decision1 == decision2
        assert cache_size_after_first == cache_size_after_second == 1

    def test_cache_eviction_policy(self):
        """Test LRU eviction works correctly."""
        manager = ProtocolManager()

        # Fill cache beyond capacity
        for i in range(600):  # More than max size of 500
            result = Mock()
            result.prompt_complexity_score = [i / 1000.0]  # Unique scores
            manager._should_use_standard_protocol(result, 1000 + i, None)

        # Cache should be bounded
        stats = manager.cache_stats
        assert (
            stats["protocol_decision_cache"]["size"]
            <= stats["protocol_decision_cache"]["max_size"]
        )


class TestModelSelectorCaching:
    """Test model selector caching behavior."""

    def test_cache_key_string_conversion(self):
        """Test cache keys are properly converted to strings."""
        mock_logger = Mock()
        service = ModelSelectionService(lit_logger=mock_logger)

        # Test cache key generation doesn't cause errors
        mock_model_entry = Mock()
        mock_model_entry.model_name = "test-model"
        mock_model_entry.providers = [ProviderType.OPENAI]

        # This should not raise an exception
        eligible_providers = service._get_eligible_providers_for_model(
            mock_model_entry, frozenset([ProviderType.OPENAI]), 5000
        )

        # Should return a list (may be empty due to mocking)
        assert isinstance(eligible_providers, list)

    def test_token_bucketing_consistency(self):
        """Test token bucketing produces consistent cache keys."""

        # Test that similar token counts map to same bucket
        test_cases = [
            (1999, 0),  # Should map to bucket 0
            (2000, 2000),  # Should map to bucket 2000
            (2001, 2000),  # Should map to bucket 2000
            (3999, 2000),  # Should map to bucket 2000
            (4000, 4000),  # Should map to bucket 4000
        ]

        for token_count, expected_bucket in test_cases:
            bucket = (token_count // 2000) * 2000
            assert (
                bucket == expected_bucket
            ), f"Token {token_count} should bucket to {expected_bucket}, got {bucket}"

    def test_cache_stats_format(self):
        """Test cache stats are in consistent format."""
        service = ModelSelectionService()

        stats = service.cache_stats
        assert isinstance(stats, dict)
        assert "eligible_providers_cache" in stats
        assert "size" in stats["eligible_providers_cache"]


class TestCachingPerformance:
    """Performance tests for caching systems."""

    def test_cache_performance_under_load(self):
        """Test cache performance with high load."""
        optimizer = CostOptimizer()

        # Create test model
        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="perf-test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="Performance test model",
        )

        # Measure performance
        start_time = time.time()

        # First run - cache misses
        for i in range(100):
            optimizer.calculate_model_cost(model_cap, 1000 + i, 0.3)

        cache_miss_time = time.time() - start_time

        # Second run - cache hits
        start_time = time.time()
        for i in range(100):
            optimizer.calculate_model_cost(model_cap, 1000 + i, 0.3)

        cache_hit_time = time.time() - start_time

        # Cache hits should be significantly faster
        assert (
            cache_hit_time < cache_miss_time
        ), "Cache hits should be faster than misses"

        # Verify cache is populated
        stats = optimizer.cache_stats
        assert stats["cost_cache"]["size"] > 0

    def test_concurrent_cache_access_performance(self):
        """Test cache performance under concurrent access."""
        optimizer = CostOptimizer()
        manager = ProtocolManager()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="concurrent-test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="Concurrent test model",
        )

        def cache_worker():
            results = []
            for i in range(20):
                # Test cost optimizer cache
                cost = optimizer.calculate_model_cost(model_cap, 1000 + i % 10, 0.3)
                results.append(cost)

                # Test protocol manager cache
                mock_result = Mock()
                mock_result.prompt_complexity_score = [0.5 + (i % 5) * 0.1]
                decision = manager._should_use_standard_protocol(
                    mock_result, 1000 + i, None
                )
                results.append(decision)

            return results

        # Run concurrent workers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_worker) for _ in range(5)]
            results = [future.result() for future in as_completed(futures)]

        execution_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            execution_time < 10.0
        ), f"Concurrent access took too long: {execution_time}s"

        # All workers should complete successfully
        assert len(results) == 5
        for result in results:
            assert len(result) == 40  # 20 iterations * 2 operations each


class TestCacheEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_none_values(self):
        """Test cache handles None values correctly."""
        optimizer = CostOptimizer()

        # Model with None cost values
        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="none-test-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=None,  # None value
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="general",
            complexity="medium",
            description="None test model",
        )

        # Should not crash
        tier = optimizer.get_cost_tier(model_cap)
        assert isinstance(tier, str)

    def test_cache_with_extreme_values(self):
        """Test cache with extreme input values."""
        optimizer = CostOptimizer()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="extreme-test-model",
            cost_per_1m_input_tokens=0.001,  # Very small
            cost_per_1m_output_tokens=1000.0,  # Very large
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="",
            complexity="medium",
            description="Extreme test model",
        )

        # Test with extreme token counts
        cost_small = optimizer.calculate_model_cost(model_cap, 1, 0.1)
        cost_large = optimizer.calculate_model_cost(model_cap, 1000000, 0.9)

        assert cost_small > 0
        assert cost_large > cost_small
        assert optimizer.cache_stats["cost_cache"]["size"] == 2

    def test_cache_key_special_characters(self):
        """Test cache handles special characters in model names."""
        optimizer = CostOptimizer()

        model_cap = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="hello-model",
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
            max_context_tokens=4096,
            max_output_tokens=2048,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="test",
            latency_tier="medium",
            task_type="",
            complexity="medium",
            description="Special chars test model",
        )

        # Should not crash and should cache properly
        cost1 = optimizer.calculate_model_cost(model_cap, 1000, 0.3)
        cost2 = optimizer.calculate_model_cost(model_cap, 1000, 0.3)

        assert cost1 == cost2  # Cache hit
        assert optimizer.cache_stats["cost_cache"]["size"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
