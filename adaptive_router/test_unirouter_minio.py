#!/usr/bin/env python3
"""Test UniRouter with MinIO storage - Local verification script."""

import logging
from typing import Any, Dict, List

from adaptive_router.models.llm_core_models import ModelSelectionRequest
from adaptive_router.services.unirouter_service import UniRouterService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_unirouter_loading():
    """Test that UniRouter loads successfully from MinIO."""
    logger.info("=" * 60)
    logger.info("Testing UniRouter with MinIO Storage")
    logger.info("=" * 60)

    try:
        # Initialize UniRouterService (will load from MinIO)
        logger.info("\n1. Initializing UniRouterService from MinIO...")
        service = UniRouterService()

        logger.info("✅ UniRouterService initialized successfully!")
        logger.info(f"   - Models loaded: {len(service.get_supported_models())}")
        logger.info(f"   - Supported models: {service.get_supported_models()}")

        # Get cluster info
        logger.info("\n2. Checking cluster information...")
        cluster_info = service.get_cluster_info()
        logger.info("✅ Cluster info retrieved:")
        logger.info(f"   - Number of clusters: {cluster_info['n_clusters']}")
        logger.info(
            f"   - Embedding model: {cluster_info['embedding_model'][:50]}..."
        )  # Truncate long model name
        logger.info(
            f"   - Lambda range: [{cluster_info['lambda_min']}, {cluster_info['lambda_max']}]"
        )
        logger.info(
            f"   - Default cost preference: {cluster_info['default_cost_preference']}"
        )

        return service

    except Exception as e:
        logger.error(f"❌ Failed to initialize UniRouterService: {e}", exc_info=True)
        return None


def test_routing(service: UniRouterService):
    """Test model routing with different prompts."""
    if not service:
        logger.error("Cannot test routing - service initialization failed")
        return

    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Routing")
    logger.info("=" * 60)

    test_cases: List[Dict[str, Any]] = [
        {
            "prompt": "Write a Python function to sort a list of numbers",
            "cost_bias": 0.5,
            "description": "Coding task with balanced cost preference",
        },
        {
            "prompt": "Explain quantum computing in simple terms",
            "cost_bias": 0.8,
            "description": "Explanation task with quality preference",
        },
        {
            "prompt": "What is 2+2?",
            "cost_bias": 0.2,
            "description": "Simple math with cost preference",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{i}. {test_case['description']}")
        logger.info(f"   Prompt: \"{test_case['prompt']}\"")
        logger.info(f"   Cost bias: {test_case['cost_bias']}")

        try:
            # Create request
            prompt = str(test_case["prompt"])
            cost_bias = float(test_case["cost_bias"])
            request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)

            # Select model
            response = service.select_model(request)

            logger.info(f"✅ Selected model: {response.provider}/{response.model}")
            logger.info(
                f"   Alternatives: {[f'{alt.provider}/{alt.model}' for alt in response.alternatives[:3]]}"
            )

        except Exception as e:
            logger.error(f"❌ Routing failed: {e}", exc_info=True)


def main():
    """Main test function."""
    # Test loading
    service = test_unirouter_loading()

    # Test routing
    if service:
        test_routing(service)

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
