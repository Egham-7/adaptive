#!/usr/bin/env python3
"""Retry failed questions with longer timeout and update predictions.

This script re-evaluates specific questions that timed out during evaluation,
uses a longer timeout (120s instead of 60s), and updates the prediction files
with the correct answers.

Usage:
    uv run python scripts/models/retry.py \\
        --model-predictions openai:gpt-5-codex --question-indices 149 \\
        --provider openai --model gpt-5-codex --api-key $OPENAI_API_KEY
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
ADAPTIVE_ROUTER_DIR = Path(__file__).parent.parent.parent / "adaptive_router"
VALIDATION_FILE = (
    ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "validation" / "validation.json"
)
PREDICTIONS_DIR = ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "predictions"


class MultipleChoiceAnswer(BaseModel):
    """Structured output for multiple choice answer."""

    answer: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct answer (A, B, C, or D)"
    )


def is_reasoning_model(model: str) -> bool:
    """Check if a model uses the reasoning endpoint (/v1/responses).

    Args:
        model: Model name

    Returns:
        True if model uses reasoning endpoint, False otherwise
    """
    reasoning_prefixes = ["gpt-5", "o1-", "o3-"]
    return any(model.startswith(prefix) for prefix in reasoning_prefixes)


def load_validation_questions() -> List[Dict[str, Any]]:
    """Load validation questions from JSON."""
    logger.info(f"Loading validation questions from {VALIDATION_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}\\n"
            f"Run Phase 1 setup first to copy validation data."
        )

    with open(VALIDATION_FILE) as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} validation questions")
    return questions


def create_prompt(question: Dict[str, Any]) -> str:
    """Create a prompt for the LLM."""
    q = question["question"]
    choices = question["choices"]

    # Format choices (choices is a list)
    labels = ["A", "B", "C", "D"]
    choices_text = "\\n".join(
        [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
    )

    prompt = f"""Answer the following multiple choice question.

Question: {q}

{choices_text}

Select the letter of the correct answer."""

    return prompt


def create_llm_client(
    model: str, api_key: str, is_reasoning: bool = False, timeout: int = 120
):
    """Create LangChain LLM client for OpenAI.

    Args:
        model: Model name
        api_key: API key for OpenAI
        is_reasoning: Whether this is a reasoning model (gpt-5, o1, o3)
        timeout: Request timeout in seconds (default: 120)

    Returns:
        LangChain chat model instance
    """
    kwargs = {
        "model": model,
        "temperature": 0.0,
        "api_key": api_key,
        "timeout": timeout,
    }
    if is_reasoning:
        kwargs["output_version"] = "responses/v1"

    return ChatOpenAI(**kwargs)


def call_llm_with_structured_output(llm: Any, prompt: str, max_retries: int = 3) -> str:
    """Call LLM with structured output and retries.

    Args:
        llm: LangChain chat model
        prompt: Question prompt
        max_retries: Maximum number of retry attempts

    Returns:
        Predicted answer (A, B, C, or D)
    """
    structured_llm = llm.with_structured_output(MultipleChoiceAnswer)

    for attempt in range(max_retries):
        try:
            logger.info(f"  Attempt {attempt + 1}/{max_retries}...")
            response = structured_llm.invoke(prompt)
            logger.info(f"  ✅ Got answer: {response.answer}")
            return response.answer

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(f"  ❌ API call failed after {max_retries} attempts: {e}")
                return "A"  # Default answer on failure

    return "A"


def load_predictions(model_id: str) -> Dict[str, str]:
    """Load existing predictions from file.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-5-codex")

    Returns:
        Dictionary mapping question_id to predicted answer
    """
    filename = model_id.replace(":", "_") + "_predictions.json"
    predictions_file = PREDICTIONS_DIR / filename

    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    with open(predictions_file) as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {predictions_file}")
    return predictions


def save_predictions(model_id: str, predictions: Dict[str, str]):
    """Save updated predictions to file.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-5-codex")
        predictions: Dictionary mapping question_id to predicted answer
    """
    filename = model_id.replace(":", "_") + "_predictions.json"
    predictions_file = PREDICTIONS_DIR / filename

    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"✅ Updated predictions saved to {predictions_file}")


def retry_failed_questions(
    model_id: str,
    provider: str,
    model: str,
    api_key: str,
    question_indices: List[int],
    timeout: int = 120,
):
    """Retry failed questions and update predictions.

    Args:
        model_id: Model identifier for predictions file (e.g., "openai:gpt-5-codex")
        provider: Provider name (openai)
        model: Model name (e.g., gpt-5-codex)
        api_key: API key
        question_indices: List of question indices (0-based) to retry
        timeout: Timeout in seconds for API calls
    """
    logger.info("=" * 80)
    logger.info(f"Retrying Failed Questions for {model_id}")
    logger.info("=" * 80)

    # Load validation questions
    all_questions = load_validation_questions()

    # Load existing predictions
    predictions = load_predictions(model_id)

    # Check if this is a reasoning model
    is_reasoning = provider == "openai" and is_reasoning_model(model)

    if not is_reasoning:
        logger.error("This script only supports reasoning models (gpt-5-*, o1-*, o3-*)")
        sys.exit(1)

    logger.info(
        f"Using reasoning endpoint (output_version='responses/v1') with {timeout}s timeout"
    )
    logger.info(f"Questions to retry: {question_indices}")

    # Create LLM client
    llm = create_llm_client(model, api_key, is_reasoning=is_reasoning, timeout=timeout)

    # Retry each failed question
    for idx in question_indices:
        if idx < 0 or idx >= len(all_questions):
            logger.error(
                f"Invalid question index: {idx} (valid range: 0-{len(all_questions)-1})"
            )
            continue

        question = all_questions[idx]
        question_id = question["question_id"]
        correct_answer = question["answer"]

        logger.info(f"\n📝 Retrying question {idx} (ID: {question_id})")
        logger.info(f"   Correct answer: {correct_answer}")

        # Create prompt
        prompt = create_prompt(question)

        # Call API with LangChain structured output
        predicted_answer = call_llm_with_structured_output(llm, prompt)

        # Update predictions
        old_answer = predictions.get(question_id, "?")
        predictions[question_id] = predicted_answer

        logger.info(f"   Old prediction: {old_answer}")
        logger.info(f"   New prediction: {predicted_answer}")
        logger.info(
            f"   {'✅ CORRECT' if predicted_answer == correct_answer else '❌ INCORRECT'}"
        )

    # Save updated predictions
    save_predictions(model_id, predictions)

    # Calculate and display updated accuracy
    all_correct = sum(
        1
        for q in all_questions[:160]
        if predictions.get(q["question_id"]) == q["answer"]
    )
    total = min(160, len(all_questions))
    accuracy = all_correct / total

    logger.info("\\n" + "=" * 80)
    logger.info("Updated Results:")
    logger.info(f"  Correct: {all_correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Error Rate: {1 - accuracy:.3f}")
    logger.info("=" * 80)


def main():
    """Main retry pipeline."""
    parser = argparse.ArgumentParser(
        description="Retry failed questions with longer timeout"
    )
    parser.add_argument(
        "--model-predictions",
        type=str,
        required=True,
        help="Model ID for predictions file (e.g., openai:gpt-5-codex)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-5-codex)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key",
    )
    parser.add_argument(
        "--question-indices",
        type=int,
        nargs="+",
        required=True,
        help="Question indices (0-based) to retry (e.g., 129 149)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for API calls (default: 120)",
    )

    args = parser.parse_args()

    try:
        retry_failed_questions(
            model_id=args.model_predictions,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            question_indices=args.question_indices,
            timeout=args.timeout,
        )

        logger.info("\\n✅ Retry complete!")

    except Exception as e:
        logger.error(f"Retry failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
