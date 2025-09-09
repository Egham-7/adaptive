"""Simplified Modal API client for NVIDIA prompt classifier.

This module provides a streamlined client to communicate with the NVIDIA prompt classifier
deployed on Modal. It handles JWT authentication, HTTP requests, and retries.
Modal returns complete classification results that are used directly.
"""

import asyncio
from datetime import datetime, timedelta
import logging
import os
import time
from typing import Any

import httpx
from jose import jwt
from pydantic import BaseModel

from adaptive_ai.models.llm_classification_models import ClassificationResult

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


class ClassifyRequest(BaseModel):
    """Request model for Modal classification API."""

    prompts: list[str]


class ModalPromptClassifier:
    """Simplified Modal API client for NVIDIA prompt classifier.

    This client makes HTTP requests to the Modal-deployed NVIDIA model and
    returns the complete classification results directly from Modal.

    Features:
    - JWT authentication with automatic token generation
    - Retry logic with exponential backoff
    - Direct use of Modal's complete classification results
    - Health check functionality
    """

    def __init__(self, lit_logger: Any = None):
        """Initialize Modal API client.

        Args:
            lit_logger: Optional LitServe logger (maintained for compatibility)
        """
        self.config = ModalConfig()
        self.lit_logger = lit_logger
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

        if self.lit_logger:
            self.lit_logger.log(
                "modal_classification_batch_start", {"batch_size": len(prompts)}
            )

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

            # Parse response - Modal returns complete results as flat lists
            response_data = response.json()

            # Convert Modal's flat structure to individual ClassificationResult objects
            results = []
            num_prompts = len(prompts)

            for i in range(num_prompts):
                # Extract single-prompt data from Modal's batch results
                single_result = {}
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > i:
                        # Wrap the single value in a list for ClassificationResult consistency
                        single_result[key] = [value[i]]
                    else:
                        # Handle non-list values or insufficient data
                        single_result[key] = (
                            [value] if not isinstance(value, list) else value
                        )

                # Ensure required fields exist for our schema
                single_result.setdefault(
                    "task_type",
                    [single_result.get("task_type_1", ["Other"])[0]],
                )
                single_result.setdefault(
                    "complexity_score",
                    [single_result.get("prompt_complexity_score", [0.5])[0]],
                )
                single_result.setdefault("domain", ["General"])

                results.append(ClassificationResult(**single_result))

            if self.lit_logger:
                self.lit_logger.log(
                    "modal_classification_batch_complete", {"batch_size": len(results)}
                )

            logger.info(f"Successfully classified {len(results)} prompts")
            return results

        except Exception as e:
            logger.error(f"Modal classification failed: {e}")
            if self.lit_logger:
                self.lit_logger.log("modal_classification_error", {"error": str(e)})

            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"Modal prompt classification failed: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Check Modal service health.

        Returns:
            Health status information from Modal service
        """
        try:
            response = self._make_request_with_retry(
                method="GET", url=f"{self.config.modal_url}/health"
            )
            return response.json()  # type: ignore[no-any-return]
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

        if self.lit_logger:
            self.lit_logger.log(
                "modal_classification_batch_start", {"batch_size": len(prompts)}
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

            # Parse response - Modal returns complete results as flat lists
            response_data = response.json()

            # Convert Modal's flat structure to individual ClassificationResult objects
            results = []
            num_prompts = len(prompts)

            for i in range(num_prompts):
                # Extract single-prompt data from Modal's batch results
                single_result = {}
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > i:
                        # Wrap the single value in a list for ClassificationResult consistency
                        single_result[key] = [value[i]]
                    else:
                        # Handle non-list values or insufficient data
                        single_result[key] = (
                            [value] if not isinstance(value, list) else value
                        )

                # Ensure required fields exist for our schema
                single_result.setdefault(
                    "task_type",
                    [single_result.get("task_type_1", ["Other"])[0]],
                )
                single_result.setdefault(
                    "complexity_score",
                    [single_result.get("prompt_complexity_score", [0.5])[0]],
                )
                single_result.setdefault("domain", ["General"])

                results.append(ClassificationResult(**single_result))

            if self.lit_logger:
                self.lit_logger.log(
                    "modal_classification_batch_complete", {"batch_size": len(results)}
                )

            logger.info(f"Successfully classified {len(results)} prompts (async)")
            return results

        except Exception as e:
            logger.error(f"Modal classification failed (async): {e}")
            if self.lit_logger:
                self.lit_logger.log("modal_classification_error", {"error": str(e)})

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


def get_modal_prompt_classifier(lit_logger: Any = None) -> ModalPromptClassifier:
    """Get Modal prompt classifier instance.

    Args:
        lit_logger: Optional LitServe logger

    Returns:
        ModalPromptClassifier instance
    """
    return ModalPromptClassifier(lit_logger=lit_logger)
