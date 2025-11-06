#!/usr/bin/env python3
"""Base provider evaluator abstraction for UniRouter model evaluation.

This module provides the abstract base class for provider-specific evaluators,
ensuring consistent interface and behavior across all LLM providers.

Following best practices from adaptive-backend's service architecture.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class MultipleChoiceAnswer(BaseModel):
    """Structured output model for multiple choice answers.

    Attributes:
        answer: The letter of the correct answer (A, B, C, or D)
    """

    answer: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct answer (A, B, C, or D)"
    )


class EvaluationError(Exception):
    """Base exception for evaluation errors."""

    pass


class APIError(EvaluationError):
    """Exception raised when API calls fail."""

    pass


class BaseEvaluator(ABC):
    """Abstract base class for provider-specific evaluators.

    This class defines the interface that all provider evaluators must implement,
    ensuring consistent behavior across different LLM providers.

    Attributes:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-5-sonnet")
        api_key: API key for authentication
        rate_limit_delay: Delay in seconds between API calls
        max_retries: Maximum number of retry attempts for failed API calls
        client: Provider-specific LLM client instance
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the evaluator.

        Args:
            provider: Provider name (e.g., "anthropic", "gemini")
            model: Model name (e.g., "claude-3-5-sonnet", "gemini-2.5-pro")
            api_key: API key for authentication
            rate_limit_delay: Delay between API calls in seconds (default: 0.5)
            max_retries: Maximum retry attempts for failed requests (default: 3)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.client: Any = None

    @abstractmethod
    def create_client(self) -> Any:
        """Create and return the LLM client for this provider.

        Each provider implementation must create its specific client
        (e.g., ChatAnthropic, ChatGoogleGenerativeAI, etc.).

        Returns:
            Provider-specific LLM client instance

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_client()")

    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt and return the answer.

        This method handles provider-specific API calling logic, including
        structured output formatting and error handling.

        Args:
            prompt: The formatted prompt to send to the LLM

        Returns:
            Predicted answer as a single letter (A, B, C, or D)

        Raises:
            APIError: If the API call fails after max_retries attempts
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement call_llm()")

    def create_prompt(self, question: Dict[str, Any]) -> str:
        """Create a formatted prompt from a question dictionary.

        Args:
            question: Question dictionary with keys:
                - question: The question text
                - choices: List of answer choices
                - answer: Correct answer letter (not used in prompt)

        Returns:
            Formatted prompt string ready for LLM consumption
        """
        q = question["question"]
        choices = question["choices"]

        # Format choices with letter labels
        labels = ["A", "B", "C", "D"]
        choices_text = "\n".join(
            [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
        )

        prompt = f"""Answer the following multiple choice question.

Question: {q}

{choices_text}

Select the letter of the correct answer."""

        return prompt

    def evaluate(
        self, questions: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], float, float]:
        """Evaluate the model on a list of validation questions.

        Args:
            questions: List of question dictionaries, each containing:
                - question_id: Unique identifier for the question
                - question: The question text
                - choices: List of answer choices
                - answer: Correct answer letter

        Returns:
            Tuple containing:
                - predictions: Dict mapping question_id to predicted answer
                - accuracy: Float between 0 and 1
                - error_rate: Float between 0 and 1 (1 - accuracy)

        Raises:
            EvaluationError: If evaluation fails
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {self.provider}/{self.model}")
        logger.info(f"{'='*80}")

        # Initialize client if not already created
        if self.client is None:
            try:
                self.client = self.create_client()
                logger.info(f"Created {self.provider} client successfully")
            except Exception as e:
                raise EvaluationError(f"Failed to create client: {e}") from e

        predictions: Dict[str, str] = {}
        correct = 0

        # Evaluate each question
        for question in tqdm(
            questions, desc=f"Evaluating {self.provider}/{self.model}"
        ):
            question_id = question["question_id"]
            correct_answer = question["answer"]

            # Create and send prompt
            prompt = self.create_prompt(question)
            try:
                predicted_answer = self.call_llm(prompt)
            except Exception as e:
                logger.error(
                    f"Failed to get prediction for question {question_id}: {e}"
                )
                # Re-raise the error instead of silently defaulting to "A"
                raise EvaluationError(
                    f"Failed to get prediction for question {question_id}: {e}"
                ) from e

            # Store prediction
            predictions[question_id] = predicted_answer

            # Track accuracy
            if predicted_answer == correct_answer:
                correct += 1

            # Rate limiting - be respectful to APIs
            time.sleep(self.rate_limit_delay)

        # Calculate metrics
        total = len(questions)
        accuracy = correct / total if total > 0 else 0.0
        error_rate = 1.0 - accuracy

        # Log results
        logger.info(f"\n{self.provider}/{self.model} Results:")
        logger.info(f"  Correct: {correct}/{total}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Error Rate: {error_rate:.3f}")

        return predictions, accuracy, error_rate

    def save_predictions(self, predictions: Dict[str, str], output_dir: Path) -> Path:
        """Save predictions to JSON file.

        Args:
            predictions: Dictionary mapping question_id to predicted answer
            output_dir: Directory to save predictions file

        Returns:
            Path to the saved predictions file

        Raises:
            IOError: If file cannot be written
        """
        # Create model ID and filename
        model_id = f"{self.provider}/{self.model}"
        filename = model_id.replace(":", "_").replace("/", "_") + "_predictions.json"
        output_file = output_dir / filename

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write predictions
        try:
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"✅ Saved predictions to {output_file}")
        except IOError as e:
            raise IOError(f"Failed to save predictions: {e}") from e

        return output_file
