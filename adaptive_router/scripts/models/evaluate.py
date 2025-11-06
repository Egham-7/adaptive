#!/usr/bin/env python3
"""Async evaluation with semaphore-based rate limiting for fast parallel processing.

This script provides 10-100x speedup over sequential evaluation by using:
- asyncio for non-blocking I/O
- Semaphore-based rate limiting for controlled concurrency
- LangChain's ainvoke() for async API calls

Usage:
    # Test with 10 questions
    uv run python scripts/models/evaluate.py --provider zai --model glm-4.6 --api-key $ZAI_API_KEY --max-questions 10

    # Full 600 questions with 20 concurrent requests
    uv run python scripts/models/evaluate.py --provider zai --model glm-4.6 --api-key $ZAI_API_KEY --max-concurrent 20
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm as atqdm

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


def create_llm_client(provider: str, model: str, api_key: str):
    """Create LangChain LLM client based on provider.

    Args:
        provider: Provider name (openai, anthropic, gemini, grok, groq, deepseek, zai)
        model: Model name
        api_key: API key for the provider

    Returns:
        LangChain chat model instance
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
            timeout=60.0,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=0.0,
            api_key=api_key,
            timeout=60.0,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=api_key,
            timeout=60.0,
        )
    elif provider in ["grok", "xai"]:
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=60.0,
        )
    elif provider == "groq":
        return ChatGroq(
            model=model,
            temperature=0.0,
            api_key=api_key,
            timeout=60.0,
        )
    elif provider == "deepseek":
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=60.0,
        )
    elif provider == "zai":
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
            base_url="https://api.z.ai/api/paas/v4",
            timeout=60.0,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: openai, anthropic, gemini, grok, groq, deepseek, zai"
        )


def extract_answer_from_text(text: str) -> str:
    """Extract answer letter from text response.

    Args:
        text: Text response from LLM

    Returns:
        Extracted answer (A, B, C, or D)
    """
    # Try to find answer patterns
    patterns = [
        r"\b([ABCD])\b",  # Single letter
        r"answer\s*:?\s*([ABCD])",  # "answer: A" or "Answer A"
        r"correct\s+answer\s*:?\s*([ABCD])",  # "correct answer: B"
        r"\(([ABCD])\)",  # (A)
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Default to first letter found
    for char in text.upper():
        if char in ["A", "B", "C", "D"]:
            return char

    return "A"  # Default if nothing found


def create_prompt(question: Dict[str, Any]) -> str:
    """Create a prompt for the LLM."""
    q = question["question"]
    choices = question["choices"]

    # Format choices
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
    )

    prompt = f"""Answer the following multiple choice question.

Question: {q}

{choices_text}

Select the letter of the correct answer."""

    return prompt


async def call_llm_async(
    llm: Any,
    question: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """Call LLM asynchronously with rate limiting.

    Args:
        llm: LangChain chat model
        question: Question dict
        semaphore: Asyncio semaphore for rate limiting
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (question_id, predicted_answer)
    """
    question_id = question["question_id"]
    prompt = create_prompt(question)

    async with semaphore:  # Limit concurrent requests
        for attempt in range(max_retries):
            try:
                # Use ainvoke for async call
                response = await llm.ainvoke(prompt)
                text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                answer = extract_answer_from_text(text)
                return question_id, answer

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Retry {attempt+1} for {question_id}: {str(e)[:100]}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed {question_id} after {max_retries} attempts: {str(e)[:100]}"
                    )
                    return question_id, "A"  # Default on failure

    return question_id, "A"


async def evaluate_async(
    provider: str,
    model: str,
    api_key: str,
    questions: List[Dict[str, Any]],
    max_concurrent: int = 20,
) -> Dict[str, str]:
    """Evaluate model asynchronously with controlled concurrency.

    Args:
        provider: Provider name
        model: Model name
        api_key: API key
        questions: List of questions
        max_concurrent: Maximum concurrent requests

    Returns:
        Dictionary mapping question_id to predicted answer
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Async Evaluation: {provider}/{model}")
    logger.info(f"  Questions: {len(questions)}")
    logger.info(f"  Max concurrent requests: {max_concurrent}")
    logger.info(f"{'='*80}\n")

    # Create client
    llm = create_llm_client(provider, model, api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create async tasks for all questions
    tasks = [call_llm_async(llm, question, semaphore) for question in questions]

    # Run all tasks with progress bar
    start_time = time.time()
    results = []

    # Use asyncio.gather with progress tracking
    for coro in atqdm.as_completed(
        tasks, total=len(tasks), desc=f"Evaluating {provider}/{model}"
    ):
        result = await coro
        results.append(result)

    elapsed = time.time() - start_time

    # Convert results to predictions dict
    predictions = dict(results)

    # Calculate accuracy
    correct = sum(
        1 for q in questions if predictions.get(q["question_id"]) == q["answer"]
    )
    accuracy = correct / len(questions)
    error_rate = 1 - accuracy

    logger.info(f"\n{'='*80}")
    logger.info(f"{provider}/{model} Results:")
    logger.info(f"  Correct: {correct}/{len(questions)}")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Error Rate: {error_rate:.3f}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/len(questions):.2f}s per question)")
    logger.info(
        f"  Speedup: ~{len(questions) * 15 / elapsed:.1f}x (vs 15s/question baseline)"
    )
    logger.info(f"{'='*80}\n")

    return predictions


def load_validation_questions() -> List[Dict[str, Any]]:
    """Load validation questions from JSON."""
    logger.info(f"Loading validation questions from {VALIDATION_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}\n"
            f"Run Phase 1 setup first to copy validation data."
        )

    with open(VALIDATION_FILE) as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} validation questions")
    return questions


def save_predictions(model_id: str, predictions: Dict[str, str]):
    """Save predictions to JSON file.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o-mini")
        predictions: Dictionary mapping question_id to predicted answer
    """
    filename = model_id.replace("/", "_") + "_predictions.json"
    output_file = PREDICTIONS_DIR / filename

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"✅ Saved predictions to {output_file}")


async def main_async():
    """Main async evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Async LLM evaluation with concurrent requests"
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "anthropic", "gemini", "grok", "groq", "deepseek", "zai"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o-mini, glm-4.6)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or use environment variable)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to evaluate (for testing)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom model name for output file. If not provided, uses actual model name.",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UniRouter: Async LLM Evaluation")
    logger.info("=" * 80)

    # Get API key
    api_key = args.api_key
    if not api_key:
        # Try environment variables
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "grok": "GROK_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zai": "ZAI_API_KEY",
        }
        env_var = env_vars.get(args.provider)
        api_key = os.environ.get(env_var)

        if not api_key:
            logger.error(
                f"API key not provided. Set {env_var} environment variable or use --api-key"
            )
            sys.exit(1)

    # Load questions
    questions = load_validation_questions()

    # Limit to max_questions if specified
    if args.max_questions:
        logger.info(
            f"Limiting evaluation to first {args.max_questions} questions (out of {len(questions)} total)"
        )
        questions = questions[: args.max_questions]

    # Run async evaluation
    try:
        predictions = await evaluate_async(
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            questions=questions,
            max_concurrent=args.max_concurrent,
        )

        # Save predictions
        model_name = args.output_name if args.output_name else args.model
        model_id = f"{args.provider}/{model_name}"
        save_predictions(model_id, predictions)

        logger.info("\n" + "=" * 80)
        logger.info("✅ Async evaluation complete!")
        logger.info(
            "   Next step: Run scripts/models/profile.py to compute error rates"
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
