"""NVIDIA Prompt Classifier client for Modal API communication.

This module provides a complete client interface to communicate with the NVIDIA prompt classifier
deployed on Modal. It handles JWT authentication, HTTP requests, retries, and response processing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
import os
import time
from typing import Any, Annotated

import httpx
from jose import jwt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type alias for probability values constrained to [0.0, 1.0] range
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class ClassificationResult(BaseModel):
    """Results from prompt classification including task type and complexity metrics.

    This model contains the output from ML classifiers that analyze prompts
    to determine task types, complexity scores, and various task characteristics.
    All list fields contain results for batch processing where each index
    corresponds to a single prompt in the batch.

    Attributes:
        task_type: Primary task type classifications for each prompt (required)
        complexity_score: Overall complexity scores (0.0-1.0) (required)
        domain: Domain classifications for each prompt (required)
        task_type_1: Primary task type classifications for each prompt
        task_type_2: Secondary task type classifications for each prompt
        task_type_prob: Confidence scores for primary task types
        creativity_scope: Creativity level scores (0.0-1.0)
        reasoning: Reasoning complexity scores (0.0-1.0)
        contextual_knowledge: Required contextual knowledge scores (0.0-1.0)
        prompt_complexity_score: Overall complexity scores (0.0-1.0)
        domain_knowledge: Domain-specific knowledge requirement scores (0.0-1.0)
        number_of_few_shots: Few-shot learning requirement scores (0.0-1.0)
        no_label_reason: Confidence in classification accuracy (0.0-1.0)
        constraint_ct: Number of constraints detected in prompts (0.0-1.0)
    """

    # Required fields for compatibility with tests
    task_type: list[str] = Field(
        description="Primary task type for each prompt in batch (required)",
        examples=[["Text Generation", "Code Generation"]],
    )
    complexity_score: list[UnitFloat] = Field(
        description="Overall complexity scores (0=simple, 1=complex) (required)",
        examples=[[0.45, 0.72]],
    )
    domain: list[str] = Field(
        description="Domain classifications for each prompt (required)",
        examples=[["General", "Technical"]],
    )

    # Optional detailed fields
    task_type_1: list[str] | None = Field(
        default=None,
        description="Primary task type for each prompt in batch",
        examples=[["Text Generation", "Code Generation"]],
    )
    task_type_2: list[str] | None = Field(
        default=None,
        description="Secondary task type for each prompt in batch",
        examples=[["Summarization", "Classification"]],
    )
    task_type_prob: list[UnitFloat] | None = Field(
        default=None,
        description="Confidence scores for primary task types",
        examples=[[0.89, 0.76]],
    )
    creativity_scope: list[UnitFloat] | None = Field(
        default=None,
        description="Creativity level required for each task (0=analytical, 1=creative)",
        examples=[[0.2, 0.8]],
    )
    reasoning: list[UnitFloat] | None = Field(
        default=None,
        description="Reasoning complexity required (0=simple, 1=complex)",
        examples=[[0.7, 0.4]],
    )
    contextual_knowledge: list[UnitFloat] | None = Field(
        default=None,
        description="Context knowledge requirement (0=none, 1=extensive)",
        examples=[[0.3, 0.6]],
    )
    prompt_complexity_score: list[UnitFloat] | None = Field(
        default=None,
        description="Overall prompt complexity (0=simple, 1=complex)",
        examples=[[0.45, 0.72]],
    )
    domain_knowledge: list[UnitFloat] | None = Field(
        default=None,
        description="Domain-specific knowledge requirement (0=general, 1=specialist)",
        examples=[[0.1, 0.9]],
    )
    number_of_few_shots: list[int] | None = Field(
        default=None,
        description="Few-shot learning requirement (number of examples needed)",
        examples=[[0, 3]],
    )
    no_label_reason: list[UnitFloat] | None = Field(
        default=None,
        description="Confidence in classification accuracy (0=low, 1=high)",
        examples=[[0.9, 0.85]],
    )
    constraint_ct: list[UnitFloat] | None = Field(
        default=None,
        description="Constraint complexity detected (0=none, 1=many constraints)",
        examples=[[0.2, 0.5]],
    )


class ModalConfig:
    """Configuration for Modal API client with performance optimizations."""

    def __init__(self) -> None:
        self.modal_url = os.environ.get("MODAL_CLASSIFIER_URL", "url")
        self.jwt_secret = os.environ.get("JWT_SECRET")
        if not self.jwt_secret:
            logger.warning("JWT_SECRET not set, Modal authentication will fail")

        # Increased timeouts for chunked processing
        self.request_timeout = int(os.environ.get("MODAL_REQUEST_TIMEOUT", "120"))
        self.max_retries = int(os.environ.get("MODAL_MAX_RETRIES", "3"))
        self.retry_delay = float(os.environ.get("MODAL_RETRY_DELAY", "1.0"))

        # Connection pooling settings
        self.max_connections = int(os.environ.get("MODAL_MAX_CONNECTIONS", "20"))
        self.max_keepalive_connections = int(
            os.environ.get("MODAL_MAX_KEEPALIVE", "10")
        )
        self.keepalive_expiry = int(os.environ.get("MODAL_KEEPALIVE_EXPIRY", "30"))

        # Chunking settings for improved performance
        self.default_chunk_size = int(os.environ.get("MODAL_CHUNK_SIZE", "10"))
        self.max_concurrent_chunks = int(os.environ.get("MODAL_MAX_CONCURRENT", "5"))


class ClassifyRequest(BaseModel):
    """Request model for Modal classification API with chunking support."""

    prompts: list[str]
    chunk_size: int | None = Field(default=None, ge=1, le=50)
    max_concurrent: int | None = Field(default=None, ge=1, le=10)


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
        """Initialize Modal API client with optimized connection pooling.

        Args:
            lit_logger: Optional LitServe logger (maintained for compatibility)
        """
        self.config = ModalConfig()
        self.lit_logger = lit_logger

        # Optimized connection limits for better performance
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )

        # Create clients with connection pooling
        self.client = httpx.Client(
            timeout=self.config.request_timeout,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
        )
        self.async_client = httpx.AsyncClient(
            timeout=self.config.request_timeout,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
        )

        self._token_cache: str | None = None
        self._token_expires_at: datetime | None = None

        logger.info(
            f"Initialized optimized Modal client for URL: {self.config.modal_url}, "
            f"max_connections: {self.config.max_connections}, "
            f"chunk_size: {self.config.default_chunk_size}"
        )

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
            # For backward compatibility, don't send the new fields yet
            # The Modal API will use its defaults (10 chunks, 5 concurrent)
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()

            logger.info(f"Sending classification request for {len(prompts)} prompts")

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
            # For backward compatibility, don't send the new fields yet
            # The Modal API will use its defaults (10 chunks, 5 concurrent)
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()

            logger.info(
                f"Sending async classification request for {len(prompts)} prompts"
            )

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
