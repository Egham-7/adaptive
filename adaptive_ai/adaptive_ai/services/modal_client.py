"""Modal API client for NVIDIA prompt classifier.

This module provides a client to communicate with the NVIDIA prompt classifier
deployed on Modal. It handles JWT authentication, HTTP requests, retries,
and maintains the same interface as the original PromptClassifier.
"""

import asyncio
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional

import httpx
from jose import jwt
from pydantic import BaseModel

from adaptive_ai.models.llm_classification_models import ClassificationResult

logger = logging.getLogger(__name__)


class ModalConfig:
    """Configuration for Modal API client."""
    
    def __init__(self):
        self.modal_url = os.environ.get(
            "MODAL_CLASSIFIER_URL", 
            "https://your-username--nvidia-prompt-classifier-serve.modal.run"
        )
        self.jwt_secret = os.environ.get("JWT_SECRET")
        if not self.jwt_secret:
            logger.warning("JWT_SECRET not set, Modal authentication will fail")
            
        self.request_timeout = int(os.environ.get("MODAL_REQUEST_TIMEOUT", "30"))
        self.max_retries = int(os.environ.get("MODAL_MAX_RETRIES", "3"))
        self.retry_delay = float(os.environ.get("MODAL_RETRY_DELAY", "1.0"))


class ClassifyRequest(BaseModel):
    """Request model for Modal classification API."""
    prompts: List[str]


class ModalPromptClassifier:
    """Modal API client for NVIDIA prompt classifier.
    
    This client maintains the same interface as the original PromptClassifier
    but makes HTTP requests to the Modal-deployed NVIDIA model instead of
    running the model locally.
    
    Features:
    - JWT authentication with automatic token generation
    - Retry logic with exponential backoff
    - Error handling and graceful fallbacks
    - Same interface as original PromptClassifier
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
        self._token_cache: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        logger.info(f"Initialized Modal client for URL: {self.config.modal_url}")

    def _generate_jwt_token(self) -> str:
        """Generate JWT token for Modal authentication."""
        if not self.config.jwt_secret:
            raise ValueError("JWT_SECRET environment variable is required")
            
        # Token expires in 1 hour
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        payload = {
            "sub": "adaptive_ai_service",  # Subject (service identifier)
            "user": "adaptive_ai",         # User field for compatibility
            "iat": datetime.utcnow(),      # Issued at
            "exp": expires_at,             # Expires at
            "service": "prompt_classification"
        }
        
        token = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")
        
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
            "Content-Type": "application/json"
        }

    def _make_request_with_retry(
        self, 
        method: str, 
        url: str, 
        **kwargs
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
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise
        
        # This shouldn't be reached, but just in case
        raise last_exception or Exception("Request failed after all retries")

    def classify_prompts(self, prompts: List[str]) -> List[ClassificationResult]:
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
                json=request_data.dict()
            )
            
            # Parse response
            response_data = response.json()
            
            # Convert to ClassificationResult objects
            # Modal returns a single result object with lists, we need to convert
            # to individual ClassificationResult objects for each prompt
            results = []
            num_prompts = len(prompts)
            
            for i in range(num_prompts):
                # Extract single-prompt data from batch results
                single_result = {}
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > i:
                        single_result[key] = [value[i]]  # Wrap in list for consistency
                    else:
                        single_result[key] = value
                
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

    def classify_prompts_raw(self, prompts: List[str]) -> List[List[List[float]]]:
        """Classify multiple prompts and return raw logits from Modal API.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            Raw model logits as nested lists
            
        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
            
        if self.lit_logger:
            self.lit_logger.log(
                "modal_raw_classification_batch_start", {"batch_size": len(prompts)}
            )

        logger.info(f"Getting raw logits for {len(prompts)} prompts via Modal API")
        
        try:
            # Prepare request
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()
            
            # Make API request to raw endpoint
            response = self._make_request_with_retry(
                method="POST",
                url=f"{self.config.modal_url}/classify_raw",
                headers=headers,
                json=request_data.dict()
            )
            
            # Parse response - expect raw logits
            response_data = response.json()
            
            if self.lit_logger:
                self.lit_logger.log(
                    "modal_raw_classification_batch_complete", {"batch_size": len(prompts)}
                )
                
            logger.info(f"Successfully got raw logits for {len(prompts)} prompts")
            return response_data["logits"]  # Expect {"logits": [...]} structure
            
        except Exception as e:
            logger.error(f"Modal raw classification failed: {e}")
            if self.lit_logger:
                self.lit_logger.log("modal_raw_classification_error", {"error": str(e)})
            
            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"Modal raw prompt classification failed: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Check Modal service health.
        
        Returns:
            Health status information from Modal service
        """
        try:
            response = self._make_request_with_retry(
                method="GET",
                url=f"{self.config.modal_url}/health"
            )
            return response.json()
        except Exception as e:
            logger.error(f"Modal health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def classify_prompts_async(self, prompts: List[str]) -> List[ClassificationResult]:
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
                json=request_data.dict()
            )
            
            # Parse response
            response_data = response.json()
            
            # Convert to ClassificationResult objects
            # Modal returns a single result object with lists, we need to convert
            # to individual ClassificationResult objects for each prompt
            results = []
            num_prompts = len(prompts)
            
            for i in range(num_prompts):
                # Extract single-prompt data from batch results
                single_result = {}
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > i:
                        single_result[key] = [value[i]]  # Wrap in list for consistency
                    else:
                        single_result[key] = value
                
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

    async def classify_prompts_raw_async(self, prompts: List[str]) -> List[List[List[float]]]:
        """Classify multiple prompts and return raw logits from Modal API asynchronously.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            Raw model logits as nested lists
            
        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
            
        if self.lit_logger:
            self.lit_logger.log(
                "modal_raw_classification_batch_start", {"batch_size": len(prompts)}
            )

        logger.info(f"Getting raw logits for {len(prompts)} prompts via Modal API (async)")
        
        try:
            # Prepare request
            request_data = ClassifyRequest(prompts=prompts)
            headers = self._get_auth_headers()
            
            # Make async API request to raw endpoint
            response = await self._make_request_with_retry_async(
                method="POST",
                url=f"{self.config.modal_url}/classify_raw",
                headers=headers,
                json=request_data.dict()
            )
            
            # Parse response - expect raw logits
            response_data = response.json()
            
            if self.lit_logger:
                self.lit_logger.log(
                    "modal_raw_classification_batch_complete", {"batch_size": len(prompts)}
                )
                
            logger.info(f"Successfully got raw logits for {len(prompts)} prompts (async)")
            return response_data["logits"]  # Expect {"logits": [...]} structure
            
        except Exception as e:
            logger.error(f"Modal raw classification failed (async): {e}")
            if self.lit_logger:
                self.lit_logger.log("modal_raw_classification_error", {"error": str(e)})
            
            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"Modal raw prompt classification failed: {e}") from e

    async def _make_request_with_retry_async(
        self, 
        method: str, 
        url: str, 
        **kwargs
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
                    delay = self.config.retry_delay * (2 ** attempt)
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
                method="GET",
                url=f"{self.config.modal_url}/health"
            )
            return response.json()
        except Exception as e:
            logger.error(f"Modal health check failed (async): {e}")
            return {"status": "unhealthy", "error": str(e)}

    def __del__(self):
        """Clean up HTTP clients on destruction."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'async_client'):
            # Note: async client cleanup should be done properly with await
            # This is just a fallback for when proper cleanup wasn't done
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in __del__, so just close synchronously
                    loop.create_task(self.async_client.aclose())
                else:
                    loop.run_until_complete(self.async_client.aclose())
            except:
                pass  # Best effort cleanup


def get_prompt_classifier(lit_logger: Any = None) -> ModalPromptClassifier:
    """Get Modal prompt classifier instance.
    
    This function maintains compatibility with the original interface
    while returning the Modal API client instead.
    
    Args:
        lit_logger: Optional LitServe logger
        
    Returns:
        ModalPromptClassifier instance
    """
    return ModalPromptClassifier(lit_logger=lit_logger)