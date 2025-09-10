#!/usr/bin/env python3
"""Performance test script for the improved NVIDIA classifier.

This script tests the async and chunked processing improvements.
"""

import asyncio
import time
from nvidia_classifier.client import ModalPromptClassifier


async def test_async_performance():
    """Test async performance improvements."""

    # Create test prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing to a 10-year-old",
        "What is the capital of France?",
        "Generate a creative story about a robot",
        "Summarize the benefits of renewable energy",
        "Write a SQL query to find duplicate records",
        "Describe the process of photosynthesis",
        "Create a marketing plan for a new product",
        "How do neural networks work?",
        "Write a haiku about programming",
    ] * 3  # 30 prompts total for better testing

    print(f"Testing with {len(test_prompts)} prompts")

    try:
        classifier = ModalPromptClassifier()

        # Test sync performance
        print("\n=== Sync Classification Test ===")
        start_time = time.time()
        sync_results = classifier.classify_prompts(test_prompts)
        sync_duration = time.time() - start_time

        print(
            f"Sync Results: {len(sync_results)} results in {sync_duration:.2f} seconds"
        )
        print(f"Avg per prompt: {sync_duration / len(test_prompts):.3f} seconds")

        # Test async performance
        print("\n=== Async Classification Test ===")
        start_time = time.time()
        async_results = await classifier.classify_prompts_async(test_prompts)
        async_duration = time.time() - start_time

        print(
            f"Async Results: {len(async_results)} results in {async_duration:.2f} seconds"
        )
        print(f"Avg per prompt: {async_duration / len(test_prompts):.3f} seconds")

        # Compare performance
        if async_duration < sync_duration:
            improvement = ((sync_duration - async_duration) / sync_duration) * 100
            print(f"\n✅ Async is {improvement:.1f}% faster!")
        else:
            degradation = ((async_duration - sync_duration) / sync_duration) * 100
            print(f"\n⚠️ Async is {degradation:.1f}% slower")

        # Test results consistency
        if len(sync_results) == len(async_results) == len(test_prompts):
            print("✅ Result count consistency check passed")
        else:
            print(
                f"❌ Result count mismatch: sync={len(sync_results)}, async={len(async_results)}, expected={len(test_prompts)}"
            )

        # Show sample results
        print("\n=== Sample Results ===")
        for i in range(min(3, len(sync_results))):
            result = sync_results[i]
            print(f"Prompt {i+1}: {test_prompts[i][:50]}...")
            print(f"  Task Type: {result.task_type[0] if result.task_type else 'N/A'}")
            print(
                f"  Complexity: {result.complexity_score[0] if result.complexity_score else 'N/A'}"
            )
            print(f"  Domain: {result.domain[0] if result.domain else 'N/A'}")

        # Test health check
        print("\n=== Health Check ===")
        health = classifier.health_check()
        print(f"Health Status: {health}")

        # Clean up
        await classifier.aclose()
        classifier.close()

        return {
            "sync_duration": sync_duration,
            "async_duration": async_duration,
            "prompts_count": len(test_prompts),
            "results_count": len(sync_results),
            "improvement": (
                improvement if async_duration < sync_duration else -degradation
            ),
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_connection_pooling():
    """Test connection pooling improvements."""
    print("\n=== Connection Pooling Test ===")

    # Test multiple small requests (simulates real usage)
    small_prompts = ["Hello world", "Test prompt", "Another test"]

    try:
        classifier = ModalPromptClassifier()

        start_time = time.time()
        for i in range(5):  # 5 small requests
            results = classifier.classify_prompts(small_prompts)
            print(f"Request {i+1}: {len(results)} results")

        total_time = time.time() - start_time
        print(f"Total time for 5 requests: {total_time:.2f} seconds")
        print(f"Avg per request: {total_time / 5:.2f} seconds")

        # Clean up
        classifier.close()

        return total_time

    except Exception as e:
        print(f"❌ Connection pooling test failed: {e}")
        return None


async def main():
    """Run all performance tests."""
    print("🚀 NVIDIA Classifier Performance Tests")
    print("=" * 50)

    # Test async performance
    async_results = await test_async_performance()

    # Test connection pooling
    pooling_time = test_connection_pooling()

    # Summary
    print(f"\n{'=' * 50}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'=' * 50}")

    if async_results:
        print(f"✅ Async processing: {async_results['improvement']:+.1f}% vs sync")
        print(f"✅ Processed {async_results['prompts_count']} prompts successfully")
        print(
            f"✅ Avg latency: {async_results['async_duration'] / async_results['prompts_count']:.3f}s per prompt"
        )

    if pooling_time:
        print(f"✅ Connection pooling: {pooling_time / 5:.2f}s avg per request")

    print("\n🎯 Improvements implemented:")
    print("  • Async/await processing with asyncio.gather()")
    print("  • Chunked batch processing (configurable chunk size)")
    print("  • Connection pooling with HTTP/2 support")
    print("  • Concurrent request handling with semaphores")
    print("  • Optimized Modal container scaling (20 max, 2 warm)")
    print("  • Error handling with fallback responses")


if __name__ == "__main__":
    asyncio.run(main())
