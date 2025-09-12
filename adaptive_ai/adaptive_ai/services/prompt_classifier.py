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

import asyncio
from datetime import datetime, timedelta
import logging
import os
import time
from typing import Any

import httpx
from jose import jwt

from adaptive_ai.models.llm_classification_models import (
    ClassificationResult,
    ClassifyRequest,
    SingleClassifyRequest,
)

logger = logging.getLogger(__name__)


class ModalConfig:
    """Configuration for Modal API client."""

    def __init__(self) -> None:
        self.modal_url = os.environ.get("MODAL_CLASSIFIER_URL", "url")
        self.jwt_secret = os.environ.get("JWT_SECRET")
        if not self.jwt_secret:
            logger.warning("JWT_SECRET not set, Modal authentication will fail")

        self.request_timeout = int(os.environ.get("MODAL_REQUEST_TIMEOUT", "30"))
        self.max_retries = int(os.environ.get("MODAL_MAX_RETRIES", "3"))
        self.retry_delay = float(os.environ.get("MODAL_RETRY_DELAY", "1.0"))


class PromptClassifier:
    """Simplified Modal API client for NVIDIA prompt classifier.

    This client makes HTTP requests to the Modal-deployed NVIDIA model and
    returns the complete classification results directly from Modal.

    Features:
    - JWT authentication with automatic token generation
    - Retry logic with exponential backoff
    - Direct use of Modal's complete classification results
    - Health check functionality
    """

    def __init__(self) -> None:
        """Initialize Modal API client."""
        self.config = ModalConfig()
        self.client = httpx.Client(timeout=self.config.request_timeout)
        self.async_client = httpx.AsyncClient(timeout=self.config.request_timeout)
        self._token_cache: str | None = None
        self._token_expires_at: datetime | None = None

        logger.info(f"Initialized Modal client for URL: {self.config.modal_url}")

    def _generate_jwt_token(self) -> str:
        """Generate JWT token for Modal authentication."""
        if not self.config.jwt_secret:
            raise ValueError("JWT_SECRET environment variable is required")

        # Token expires in 1 hour
        expires_at = datetime.utcnow() + timedelta(hours=1)

        payload = {
            "sub": "adaptive_ai_service",  # Subject (service identifier)
            "user": "adaptive_ai",  # User field for compatibility
            "iat": datetime.utcnow(),  # Issued at
            "exp": expires_at,  # Expires at
            "service": "prompt_classification",
        }

        token: str = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")

        # Cache token and expiration
        self._token_cache = token
        self._token_expires_at = expires_at

        logger.debug("Generated new JWT token")
        return token

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers with JWT token."""
        # Check if we need to generate/refresh token
        if (
            not self._token_cache
            or not self._token_expires_at
            or datetime.utcnow() >= self._token_expires_at - timedelta(minutes=5)
        ):
            self._generate_jwt_token()

        return {
            "Authorization": f"Bearer {self._token_cache}",
            "Content-Type": "application/json",
        }

    def _make_request_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.request(method, url, **kwargs)

                # Check for authentication errors
                if response.status_code == 401:
                    logger.warning(f"Authentication failed (attempt {attempt + 1})")
                    # Clear cached token and retry
                    self._token_cache = None
                    self._token_expires_at = None
                    if attempt < self.config.max_retries - 1:
                        kwargs["headers"] = self._get_auth_headers()
                        continue

                # Check for other HTTP errors
                response.raise_for_status()
                return response

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise

        # This shouldn't be reached, but just in case
        raise last_exception or Exception("Request failed after all retries")

    def classify_prompts(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify multiple prompts using Modal API.

        Args:
            prompts: List of prompts to classify

        Returns:
            List of classification results, one per prompt

        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        logger.info(f"Starting Modal classification batch with {len(prompts)} prompts")
        logger.info(f"Classifying {len(prompts)} prompts via Modal API")

        try:
            # Prepare request
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()

            # Make API request
            response = self._make_request_with_retry(
                method="POST",
                url=f"{self.config.modal_url}/classify",
                headers=headers,
                json=request_data.dict(),
            )

            # Parse response - Modal returns data compatible with ClassificationResult
            response_data = response.json()

            # Directly validate and convert to ClassificationResult
            results = [ClassificationResult(**response_data)]

            logger.info(
                f"Completed Modal classification batch with {len(results)} results"
            )
            logger.info(f"Successfully classified {len(results)} prompts")
            return results

        except Exception as e:
            logger.error(f"Modal classification failed: {e}")
            logger.error(f"Modal classification error details: {e!s}")

            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"Modal prompt classification failed: {e}") from e

    def classify_prompt(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt using Modal API's single endpoint.

        Args:
            prompt: Single prompt to classify

        Returns:
            Classification result for the prompt

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.info("Classifying single prompt via Modal single endpoint")

        try:
            # Use proper request model
            request_data = SingleClassifyRequest(prompt=prompt)
            headers = self._get_auth_headers()

            # Make API request to single endpoint
            response = self._make_request_with_retry(
                method="POST",
                url=f"{self.config.modal_url}/classify/single",
                headers=headers,
                json=request_data.dict(),
            )

            # Parse response - Modal returns data compatible with ClassificationResult
            response_data = response.json()

            # Directly validate and convert to ClassificationResult
            result = ClassificationResult(**response_data)
            logger.info("Successfully classified single prompt via single endpoint")
            return result

        except Exception as e:
            logger.error(f"Modal single prompt classification failed: {e}")
            raise RuntimeError(f"Modal single prompt classification failed: {e}") from e

    async def classify_prompt_async(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt using Modal API's single endpoint asynchronously.

        Args:
            prompt: Single prompt to classify

        Returns:
            Classification result for the prompt

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.info("Classifying single prompt via Modal single endpoint (async)")

        try:
            # Use proper request model
            request_data = SingleClassifyRequest(prompt=prompt)
            headers = self._get_auth_headers()

            # Make async API request to single endpoint
            response = await self._make_request_with_retry_async(
                method="POST",
                url=f"{self.config.modal_url}/classify/single",
                headers=headers,
                json=request_data.dict(),
            )

            # Parse response - Modal returns data compatible with ClassificationResult
            response_data = response.json()

            # Directly validate and convert to ClassificationResult
            result = ClassificationResult(**response_data)
            logger.info(
                "Successfully classified single prompt via single endpoint (async)"
            )
            return result

        except Exception as e:
            logger.error(f"Modal single prompt classification failed (async): {e}")
            raise RuntimeError(f"Modal single prompt classification failed: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Check Modal service health.

        Returns:
            Health status information from Modal service
        """
        try:
            response = self._make_request_with_retry(
                method="GET", url=f"{self.config.modal_url}/health"
            )
            result = response.json()
            return (
                result
                if isinstance(result, dict)
                else {"status": "unknown", "data": result}
            )
        except Exception as e:
            logger.error(f"Modal health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def classify_prompts_async(
        self, prompts: list[str]
    ) -> list[ClassificationResult]:
        """Classify multiple prompts using Modal API asynchronously.

        Args:
            prompts: List of prompts to classify

        Returns:
            List of classification results, one per prompt

        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        logger.info(
            f"Starting Modal async classification batch with {len(prompts)} prompts"
        )
        logger.info(f"Classifying {len(prompts)} prompts via Modal API (async)")

        try:
            # Prepare request
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()

            # Make async API request
            response = await self._make_request_with_retry_async(
                method="POST",
                url=f"{self.config.modal_url}/classify",
                headers=headers,
                json=request_data.dict(),
            )

            # Parse response - Modal returns data compatible with ClassificationResult
            response_data = response.json()

            # Directly validate and convert to ClassificationResult
            results = [ClassificationResult(**response_data)]

            logger.info(
                f"Completed Modal async classification batch with {len(results)} results"
            )
            logger.info(f"Successfully classified {len(results)} prompts (async)")
            return results

        except Exception as e:
            logger.error(f"Modal classification failed (async): {e}")
            logger.error(f"Modal async classification error details: {e!s}")

            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"Modal prompt classification failed: {e}") from e

    async def _make_request_with_retry_async(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Make HTTP request with retry logic asynchronously."""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_client.request(method, url, **kwargs)

                # Check for authentication errors
                if response.status_code == 401:
                    logger.warning(f"Authentication failed (attempt {attempt + 1})")
                    # Clear cached token and retry
                    self._token_cache = None
                    self._token_expires_at = None
                    if attempt < self.config.max_retries - 1:
                        kwargs["headers"] = self._get_auth_headers()
                        continue

                # Check for other HTTP errors
                response.raise_for_status()
                return response

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise

        # This shouldn't be reached, but just in case
        raise last_exception or Exception("Request failed after all retries")

    async def health_check_async(self) -> dict[str, Any]:
        """Check Modal service health asynchronously.

        Returns:
            Health status information from Modal service
        """
        try:
            response = await self._make_request_with_retry_async(
                method="GET", url=f"{self.config.modal_url}/health"
            )
            return response.json()  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Modal health check failed (async): {e}")
            return {"status": "unhealthy", "error": str(e)}

    def close(self) -> None:
        """Explicitly close HTTP clients."""
        if hasattr(self, "client"):
            self.client.close()

    async def aclose(self) -> None:
        """Explicitly close async HTTP clients."""
        if hasattr(self, "async_client"):
            await self.async_client.aclose()

    def __del__(self) -> None:
        """Clean up HTTP clients on destruction."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception as e:
                # Log cleanup errors but don't raise during destruction
                import logging

                logging.getLogger(__name__).debug(f"Client cleanup failed: {e}")


def get_prompt_classifier() -> PromptClassifier:
    """Get Modal prompt classifier instance.

    Returns:
        ModalPromptClassifier instance
    """
    return PromptClassifier()
