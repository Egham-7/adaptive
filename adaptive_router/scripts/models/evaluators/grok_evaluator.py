#!/usr/bin/env python3
"""Grok (X.AI) provider evaluator for UniRouter model evaluation.

This module provides the Grok-specific evaluator implementation using
OpenAI-compatible API with custom base URL.
"""

import logging
import time

from langchain_openai import ChatOpenAI

from .base import APIError, BaseEvaluator, MultipleChoiceAnswer

logger = logging.getLogger(__name__)


class GrokEvaluator(BaseEvaluator):
    """Evaluator for X.AI Grok models.

    Uses ChatOpenAI with custom base_url for X.AI's API endpoint.
    Grok models are OpenAI-compatible.

    Attributes:
        provider: Always "grok"
        model: Model name (e.g., "grok-4-fast-reasoning")
        client: ChatOpenAI client instance configured for X.AI
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Grok evaluator.

        Args:
            model: Model name (e.g., "grok-4-fast-reasoning", "grok-code-fast-1")
            api_key: X.AI API key
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(
            provider="grok",
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    def create_client(self) -> ChatOpenAI:
        """Create and return the ChatOpenAI client for X.AI.

        Returns:
            ChatOpenAI client instance configured for X.AI endpoint

        Raises:
            APIError: If client creation fails
        """
        try:
            client = ChatOpenAI(
                model=self.model,
                temperature=0.0,
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",  # X.AI endpoint
            )
            logger.info(f"Created Grok (X.AI) client for {self.model}")
            return client
        except Exception as e:
            raise APIError(f"Failed to create Grok client: {e}") from e

    def call_llm(self, prompt: str) -> str:
        """Call Grok API with structured output.

        Args:
            prompt: The formatted prompt to send

        Returns:
            Predicted answer letter (A, B, C, or D)

        Raises:
            APIError: If all retry attempts fail
        """
        # Create structured output client
        structured_llm = self.client.with_structured_output(MultipleChoiceAnswer)

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = structured_llm.invoke(prompt)
                return response.answer

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Grok API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Grok API call failed after {self.max_retries} attempts: {e}"
                    )
                    raise APIError(
                        f"Grok API failed after {self.max_retries} attempts"
                    ) from e

        return "A"
