#!/usr/bin/env python3
"""Gemini provider evaluator for UniRouter model evaluation.

This module provides the Gemini-specific evaluator implementation using
langchain_google_genai's ChatGoogleGenerativeAI client.
"""

import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from .base import APIError, BaseEvaluator, MultipleChoiceAnswer

logger = logging.getLogger(__name__)


class GeminiEvaluator(BaseEvaluator):
    """Evaluator for Google Gemini models.

    Uses ChatGoogleGenerativeAI from langchain_google_genai with
    structured output support.

    Attributes:
        provider: Always "gemini"
        model: Model name (e.g., "gemini-2.5-pro")
        client: ChatGoogleGenerativeAI client instance
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Gemini evaluator.

        Args:
            model: Model name (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
            api_key: Google AI API key
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(
            provider="gemini",
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    def create_client(self) -> ChatGoogleGenerativeAI:
        """Create and return the ChatGoogleGenerativeAI client.

        Returns:
            ChatGoogleGenerativeAI client instance

        Raises:
            APIError: If client creation fails
        """
        try:
            client = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.0,
                google_api_key=self.api_key,
            )
            logger.info(f"Created ChatGoogleGenerativeAI client for {self.model}")
            return client
        except Exception as e:
            raise APIError(f"Failed to create Gemini client: {e}") from e

    def call_llm(self, prompt: str) -> str:
        """Call Gemini API with structured output.

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
                        f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Gemini API call failed after {self.max_retries} attempts: {e}"
                    )
                    raise APIError(
                        f"Gemini API failed after {self.max_retries} attempts"
                    ) from e

        return "A"
