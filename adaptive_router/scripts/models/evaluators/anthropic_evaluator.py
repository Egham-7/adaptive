#!/usr/bin/env python3
"""Anthropic provider evaluator for UniRouter model evaluation.

This module provides the Anthropic-specific evaluator implementation using
langchain_anthropic's ChatAnthropic client with structured outputs.
"""

import logging
import time
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base import APIError, BaseEvaluator, MultipleChoiceAnswer

logger = logging.getLogger(__name__)


class AnthropicEvaluator(BaseEvaluator):
    """Evaluator for Anthropic Claude models.

    Uses ChatAnthropic from langchain_anthropic with structured output
    support for reliable answer extraction.

    Attributes:
        provider: Always "anthropic"
        model: Model name (e.g., "claude-sonnet-4-5-20250929")
        client: ChatAnthropic client instance
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Anthropic evaluator.

        Args:
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
            api_key: Anthropic API key
            rate_limit_delay: Delay between API calls in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(
            provider="anthropic",
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    def create_client(self) -> ChatAnthropic:
        """Create and return the ChatAnthropic client.

        Returns:
            ChatAnthropic client instance configured for this model

        Raises:
            APIError: If client creation fails
        """
        try:
            client = ChatAnthropic(
                model=self.model,
                temperature=0.0,
                api_key=self.api_key,
            )
            logger.info(f"Created ChatAnthropic client for {self.model}")
            return client
        except Exception as e:
            raise APIError(f"Failed to create Anthropic client: {e}") from e

    def call_llm(self, prompt: str) -> str:
        """Call Anthropic API with structured output.

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
                # Try structured output first
                structured_llm = self.client.with_structured_output(
                    MultipleChoiceAnswer
                )
                response = structured_llm.invoke(prompt)

                # Validate response
                if not response or not hasattr(response, "answer"):
                    logger.warning(f"Invalid structured response: {response}")
                    raise ValueError("Structured output returned invalid response")

                answer = response.answer
                if answer not in ["A", "B", "C", "D"]:
                    logger.warning(f"Invalid answer from structured output: {answer}")
                    raise ValueError(f"Invalid answer: {answer}")

                logger.debug(f"Successfully got answer: {answer}")
                return answer

            except Exception as e:
                logger.warning(
                    f"Structured output failed (attempt {attempt + 1}/{self.max_retries}): {type(e).__name__}: {e}"
                )

                # Try fallback: plain text extraction
                try:
                    logger.info("Trying fallback: plain text response...")
                    response = self.client.invoke(prompt)
                    text = response.content.strip()
                    logger.debug(f"Plain text response: {text[:100]}")

                    # Extract first letter that is A, B, C, or D
                    for char in text.upper():
                        if char in ["A", "B", "C", "D"]:
                            logger.info(f"Extracted answer from text: {char}")
                            return char

                    logger.error(f"No valid answer found in response: {text}")

                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed. Last error: {e}")
                    raise APIError(
                        f"Anthropic API failed after {self.max_retries} attempts: {e}"
                    ) from e

        # Should never reach here
        raise APIError("Unexpected: exited retry loop without return or raise")
