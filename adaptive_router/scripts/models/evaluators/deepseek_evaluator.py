#!/usr/bin/env python3
"""DeepSeek provider evaluator for UniRouter model evaluation.

This module provides the DeepSeek-specific evaluator implementation using
OpenAI-compatible API with DeepSeek's endpoint.
"""

import logging
import time

from langchain_openai import ChatOpenAI

from .base import APIError, BaseEvaluator

logger = logging.getLogger(__name__)


class DeepSeekEvaluator(BaseEvaluator):
    """Evaluator for DeepSeek models.

    Uses ChatOpenAI with custom base_url for DeepSeek's API endpoint.
    DeepSeek models are OpenAI-compatible.

    Attributes:
        provider: Always "deepseek"
        model: Model name (e.g., "deepseek-chat", "deepseek-reasoner")
        client: ChatOpenAI client instance configured for DeepSeek
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the DeepSeek evaluator.

        Args:
            model: Model name (e.g., "deepseek-chat", "deepseek-reasoner")
            api_key: DeepSeek API key
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(
            provider="deepseek",
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    def create_client(self) -> ChatOpenAI:
        """Create and return the ChatOpenAI client for DeepSeek.

        Returns:
            ChatOpenAI client instance configured for DeepSeek endpoint

        Raises:
            APIError: If client creation fails
        """
        try:
            client = ChatOpenAI(
                model=self.model,
                temperature=0.0,
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",  # DeepSeek endpoint
            )
            logger.info(f"Created DeepSeek client for {self.model}")
            return client
        except Exception as e:
            raise APIError(f"Failed to create DeepSeek client: {e}") from e

    def call_llm(self, prompt: str) -> str:
        """Call DeepSeek API without structured output.

        DeepSeek doesn't support structured output format, so we parse
        the response text manually.

        Args:
            prompt: The formatted prompt to send

        Returns:
            Predicted answer letter (A, B, C, or D)

        Raises:
            APIError: If all retry attempts fail
        """
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Call without structured output
                response = self.client.invoke(prompt)

                # Extract answer from response text
                response_text = response.content.strip().upper()

                # Try to find the answer letter
                for letter in ["A", "B", "C", "D"]:
                    if letter in response_text:
                        return letter

                # If no clear answer found, default to first letter found
                logger.warning(f"Could not parse answer from: {response_text[:100]}")
                return "A"

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"DeepSeek API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"DeepSeek API call failed after {self.max_retries} attempts: {e}"
                    )
                    raise APIError(
                        f"DeepSeek API failed after {self.max_retries} attempts"
                    ) from e

        return "A"
