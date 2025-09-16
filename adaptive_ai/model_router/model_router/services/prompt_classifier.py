import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any

import httpx
from jose import jwt

from model_router.models.llm_classification_models import (
    ClassificationResult,
    ClassifyRequest,
    ClassifyBatchRequest,
)

logger = logging.getLogger(__name__)


class ModalConfig:
    """Configuration for Modal API client."""

    def __init__(self) -> None:
        self.modal_url = os.environ.get("MODAL_CLASSIFIER_URL")
        if not self.modal_url or not self.modal_url.startswith(("http://", "https://")):
            raise ValueError("MODAL_CLASSIFIER_URL must be set to a valid http(s) URL")

        self.jwt_secret = os.environ.get("JWT_SECRET")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET must be set for Modal authentication")

        self.request_timeout = int(os.environ.get("MODAL_REQUEST_TIMEOUT", "30"))
        self.max_retries = int(os.environ.get("MODAL_MAX_RETRIES", "3"))
        self.retry_delay = float(os.environ.get("MODAL_RETRY_DELAY", "1.0"))


class PromptClassifier:
    """Modal API client for NVIDIA prompt classifier.

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
        self.async_client = httpx.AsyncClient(timeout=self.config.request_timeout)
        self._token_cache: str | None = None
        self._token_expires_at: datetime | None = None

        logger.info(
            "Initialized Modal client", extra={"modal_url": self.config.modal_url}
        )

    def _generate_jwt_token(self) -> str:
        """Generate JWT token for Modal authentication."""
        if not self.config.jwt_secret:
            raise ValueError("JWT_SECRET environment variable is required")

        # Token expires in 1 hour (UTC)
        now_utc = datetime.now(timezone.utc)
        expires_at = now_utc + timedelta(hours=1)

        payload = {
            "sub": "model_router_service",  # Subject (service identifier)
            "user": "model_router",  # User field for compatibility
            "iat": int(now_utc.timestamp()),  # Issued at (epoch seconds)
            "exp": int(expires_at.timestamp()),  # Expires at (epoch seconds)
            "service": "prompt_classification",
        }

        token: str = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")

        # Cache token and expiration
        self._token_cache = token
        self._token_expires_at = expires_at

        logger.debug(
            "Generated new JWT token", extra={"service": "prompt_classification"}
        )
        return token

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers with JWT token."""
        # Check if we need to generate/refresh token
        now_utc = datetime.now(timezone.utc)

        if (
            not self._token_cache
            or not self._token_expires_at
            or now_utc >= self._token_expires_at - timedelta(minutes=5)
        ):
            self._generate_jwt_token()

        return {
            "Authorization": f"Bearer {self._token_cache}",
            "Content-Type": "application/json",
        }

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

        logger.info(
            "Classifying single prompt via Modal",
            extra={"endpoint": "single", "method": "async"},
        )

        try:
            # Use correct model that matches prompt-task-complexity-classifier
            request_data = ClassifyRequest(prompt=prompt)
            headers = self._get_auth_headers()

            # Make async API request to single endpoint
            response = await self._make_request_with_retry_async(
                method="POST",
                url=f"{self.config.modal_url}",
                headers=headers,
                json=request_data.model_dump(),
            )

            # Parse response - Modal returns data compatible with ClassificationResult
            response_data = response.json()

            # Directly validate and convert to ClassificationResult
            result = ClassificationResult(**response_data)
            logger.info(
                "Successfully classified single prompt",
                extra={
                    "method": "async",
                    "endpoint": "single",
                    "task_type": result.task_type_1,
                    "complexity_score": result.prompt_complexity_score,
                },
            )
            return result

        except Exception as e:
            logger.error(
                "Modal single prompt classification failed",
                extra={
                    "method": "async",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeError(f"Modal single prompt classification failed: {e}") from e

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
            "Starting Modal async classification batch",
            extra={"prompts_count": len(prompts), "method": "async"},
        )

        try:
            # Use correct model that matches prompt-task-complexity-classifier
            request_data = ClassifyBatchRequest(prompts=prompts)
            headers = self._get_auth_headers()

            # Make async API request
            response = await self._make_request_with_retry_async(
                method="POST",
                url=f"{self.config.modal_url}",
                headers=headers,
                json=request_data.model_dump(),
            )

            # Parse response - Modal returns list of ClassificationResults
            response_data = response.json()

            # Convert list of dicts to list of ClassificationResult objects
            results = [
                ClassificationResult(**result_data) for result_data in response_data
            ]

            logger.info(
                "Successfully completed Modal async classification batch",
                extra={"results_count": len(results), "method": "async"},
            )
            return results

        except Exception as e:
            logger.error(
                "Modal classification failed",
                extra={
                    "method": "async",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

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
                    logger.warning(
                        "Authentication failed",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": self.config.max_retries,
                        },
                    )
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
                    "Request failed",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": self.config.max_retries,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2**attempt)
                    logger.info(
                        "Retrying request",
                        extra={"delay_seconds": delay, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "All retry attempts failed",
                        extra={
                            "max_retries": self.config.max_retries,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
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
            logger.error(
                "Modal health check failed",
                extra={
                    "method": "async",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return {"status": "unhealthy", "error": str(e)}

    async def aclose(self) -> None:
        """Explicitly close async HTTP clients."""
        await self.async_client.aclose()


def get_prompt_classifier() -> PromptClassifier:
    """Get Modal prompt classifier instance.

    Returns:
        ModalPromptClassifier instance
    """
    return PromptClassifier()
