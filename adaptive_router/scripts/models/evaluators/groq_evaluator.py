#!/usr/bin/env python3
"""Groq provider evaluator for UniRouter model evaluation.

This module provides the Groq-specific evaluator implementation using
langchain_groq's ChatGroq client with ultra-fast inference.
"""

import logging
import time

from langchain_groq import ChatGroq

from .base import APIError, BaseEvaluator, MultipleChoiceAnswer

logger = logging.getLogger(__name__)


class GroqEvaluator(BaseEvaluator):
    """Evaluator for Groq models.

    Uses ChatGroq from langchain_groq for hardware-accelerated inference.
    Groq provides ultra-fast LLM serving.

    Attributes:
        provider: Always "groq"
        model: Model name (e.g., "llama-3.3-70b-versatile")
        client: ChatGroq client instance
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Groq evaluator.

        Args:
            model: Model name (e.g., "llama-3.3-70b-versatile", "mixtral-8x7b-32768")
            api_key: Groq API key
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(
            provider="groq",
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    def create_client(self) -> ChatGroq:
        """Create and return the ChatGroq client.

        Returns:
            ChatGroq client instance

        Raises:
            APIError: If client creation fails
        """
        try:
            client = ChatGroq(
                model=self.model,
                temperature=0.0,
                api_key=self.api_key,
            )
            logger.info(f"Created ChatGroq client for {self.model}")
            return client
        except Exception as e:
            raise APIError(f"Failed to create Groq client: {e}") from e

    def call_llm(self, prompt: str) -> str:
        """Call Groq API with structured output.

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
                        f"Groq API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Groq API call failed after {self.max_retries} attempts: {e}"
                    )
                    raise APIError(
                        f"Groq API failed after {self.max_retries} attempts"
                    ) from e

        return "A"
