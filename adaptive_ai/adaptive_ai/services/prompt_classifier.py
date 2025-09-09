"""Simplified prompt classifier service using Modal GPU inference.

This module provides a streamlined prompt classification interface for the adaptive_ai service.
It uses Modal's GPU-accelerated NVIDIA model to get complete classification results directly.

Architecture:
- NVIDIA model inference: Modal GPU deployment with complete result processing
- Direct result usage: Modal returns ready-to-use classification results
- JWT authentication: Secure communication between adaptive_ai and Modal services

Benefits:
- Simplified architecture: Single source of truth for classification results
- GPU acceleration: Fast inference using Modal's T4 GPU infrastructure
- Complete results: No need for local post-processing
- Cost optimization: Pay only for GPU usage during actual model inference
"""

import logging
from typing import Any

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.services.modal_client import ModalPromptClassifier

logger = logging.getLogger(__name__)


class PromptClassifier:
    """Simplified prompt classifier using Modal's GPU-accelerated NVIDIA model.

    This class provides a clean interface to Modal's NVIDIA prompt classifier,
    returning complete classification results directly from Modal's GPU inference.

    Benefits:
    - GPU acceleration: NVIDIA model runs on Modal's T4 GPU infrastructure
    - Complete results: Modal returns ready-to-use classification data
    - Cost optimization: Pay only for GPU usage during actual model inference
    - Simple architecture: Direct use of Modal's processed results
    """

    def __init__(self, lit_logger: Any = None) -> None:
        """Initialize prompt classifier with Modal API client.

        Args:
            lit_logger: Optional LitServe logger for compatibility and metrics
        """
        self.lit_logger = lit_logger
        self._modal_client = ModalPromptClassifier(lit_logger=lit_logger)

        logger.info("Initialized PromptClassifier with Modal API client")

        # Perform health check on initialization
        try:
            health = self._modal_client.health_check()
            if health.get("status") == "healthy":
                logger.info("Modal service health check passed")
            else:
                logger.warning(f"Modal service health check failed: {health}")
        except Exception as e:
            logger.error(f"Modal service health check error: {e}")


    def classify_prompts(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify multiple prompts using Modal-deployed NVIDIA model.

        Args:
            prompts: List of prompts to classify

        Returns:
            List of classification results, one per prompt

        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        logger.info(f"Classifying {len(prompts)} prompts via Modal API")

        try:
            # Get complete results directly from Modal
            results = self._modal_client.classify_prompts(prompts)

            logger.info(f"Successfully classified {len(results)} prompts")
            return results

        except Exception as e:
            logger.error(f"Prompt classification failed: {e}")

            # Log to LitServe if available
            if self.lit_logger:
                self.lit_logger.log("prompt_classification_error", {"error": str(e)})

            # Re-raise the exception
            raise

    async def classify_prompts_async(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify multiple prompts using Modal-deployed NVIDIA model asynchronously.

        Args:
            prompts: List of prompts to classify

        Returns:
            List of classification results, one per prompt

        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        logger.info(f"Classifying {len(prompts)} prompts via Modal API (async)")

        try:
            # Get complete results directly from Modal (async)
            results = await self._modal_client.classify_prompts_async(prompts)

            logger.info(f"Successfully classified {len(results)} prompts (async)")
            return results

        except Exception as e:
            logger.error(f"Prompt classification failed (async): {e}")

            # Log to LitServe if available
            if self.lit_logger:
                self.lit_logger.log("prompt_classification_error", {"error": str(e)})

            # Re-raise the exception
            raise

    def health_check(self) -> dict[str, Any]:
        """Check Modal service health.

        Returns:
            Health status information from Modal service
        """
        return self._modal_client.health_check()


def get_prompt_classifier(lit_logger: Any = None) -> PromptClassifier:
    """Get prompt classifier instance.

    This function maintains the same interface as before but now returns
    a Modal API client-based classifier instead of the local GPU model.

    Args:
        lit_logger: Optional LitServe logger

    Returns:
        PromptClassifier instance using Modal API
    """
    return PromptClassifier(lit_logger=lit_logger)
