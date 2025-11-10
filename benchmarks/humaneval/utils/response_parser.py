"""
Response parsing utilities for extracting metrics from various API responses.

This module provides functions to extract token counts and costs from different
LLM provider API responses (Claude, GLM, Adaptive, etc.).
"""

import logging
from typing import Any

import tiktoken

from ..models.base import ResponseMetrics

logger = logging.getLogger(__name__)


class PricingCalculator:
    """
    Fallback pricing calculator for APIs that don't return cost directly.

    This should only be used as a last resort when the API doesn't provide
    cost information. Pricing may change, so direct API costs are preferred.
    """

    # Pricing per 1M tokens (input, output) - UPDATE THESE AS NEEDED
    PRICING = {
        "claude-sonnet-4-5": (3.00, 15.00),  # $3/$15 per 1M tokens
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "gpt-4": (10.00, 30.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        # Add GLM and other models as needed
        "glm-4.6": (1.00, 1.00),  # Placeholder - update with actual pricing
    }

    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on token counts.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Find matching pricing
        pricing = None
        for model_key, prices in cls.PRICING.items():
            if model_key in model.lower():
                pricing = prices
                break

        if pricing is None:
            logger.warning(f"No pricing found for model: {model}, using defaults")
            pricing = (1.00, 1.00)  # Default fallback

        input_price, output_price = pricing
        cost = (input_tokens * input_price / 1_000_000) + (
            output_tokens * output_price / 1_000_000
        )
        return cost


def parse_claude_response(response: Any, model_name: str) -> ResponseMetrics:
    """
    Parse Anthropic Claude API response to extract metrics.

    Args:
        response: Anthropic API response object
        model_name: Model identifier

    Returns:
        ResponseMetrics with extracted data
    """
    try:
        # Claude SDK returns a response object with a usage attribute
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        # Calculate cost using pricing calculator
        # (Anthropic doesn't return cost in response as of now)
        cost = PricingCalculator.calculate_cost(
            model=model_name, input_tokens=input_tokens, output_tokens=output_tokens
        )

        return ResponseMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_used=model_name,
        )

    except Exception as e:
        logger.error(f"Error parsing Claude response: {str(e)}")
        return ResponseMetrics(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model_used=model_name,
            error=str(e),
        )


def parse_glm_response(
    response_json: dict[str, Any],
    model_name: str,
    prompt: str = "",
    completion: str = "",
) -> ResponseMetrics:
    """
    Parse GLM API response to extract metrics.

    Args:
        response_json: GLM API JSON response
        model_name: Model identifier
        prompt: Original prompt (for token estimation if needed)
        completion: Generated completion (for token estimation if needed)

    Returns:
        ResponseMetrics with extracted data
    """
    try:
        # GLM typically returns usage in the response
        # Adjust based on actual GLM API response structure
        usage = response_json.get("usage", {})

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # If usage not in response, estimate with tiktoken
        if input_tokens == 0 and prompt:
            input_tokens = estimate_tokens(prompt, model_name)
        if output_tokens == 0 and completion:
            output_tokens = estimate_tokens(completion, model_name)

        # Check if GLM returns cost directly
        cost = response_json.get("cost")
        if cost is None:
            # Calculate cost using pricing calculator
            cost = PricingCalculator.calculate_cost(
                model=model_name, input_tokens=input_tokens, output_tokens=output_tokens
            )

        return ResponseMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_used=model_name,
        )

    except Exception as e:
        logger.error(f"Error parsing GLM response: {str(e)}")
        return ResponseMetrics(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model_used=model_name,
            error=str(e),
        )


def parse_adaptive_response(
    response_json: dict[str, Any], requested_models: list = None
) -> ResponseMetrics:
    """
    Parse Adaptive routing API response to extract metrics.

    Args:
        response_json: Adaptive API JSON response (OpenAI-compatible format)
        requested_models: List of models in the routing request

    Returns:
        ResponseMetrics with extracted data including actual model used
    """
    try:
        # Extract which model was actually selected
        # OpenAI format: response has 'model' field with selected model
        selected_model = response_json.get("model", "unknown")

        # Extract usage information (OpenAI format)
        usage = response_json.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Extract cost - check if Adaptive returns it in extra fields
        cost = None

        # Check for cost in various possible locations
        if "cost" in response_json:
            cost = response_json.get("cost")
        elif "usage" in response_json and "cost" in response_json["usage"]:
            cost = response_json["usage"].get("cost")
        elif "adaptive_metadata" in response_json:
            cost = response_json["adaptive_metadata"].get("cost")

        # If no cost provided, calculate based on selected model
        if cost is None:
            logger.warning(
                f"Adaptive API didn't return cost for {selected_model}, calculating..."
            )
            cost = PricingCalculator.calculate_cost(
                model=selected_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Extract routing metadata if available
        routing_overhead = 0
        if "adaptive_metadata" in response_json:
            routing_overhead = response_json["adaptive_metadata"].get(
                "routing_overhead_ms", 0
            )

        return ResponseMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_used=selected_model,  # Important: track which model was chosen
            latency_seconds=routing_overhead / 1000.0 if routing_overhead else 0,
        )

    except Exception as e:
        logger.error(f"Error parsing Adaptive response: {str(e)}")
        return ResponseMetrics(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model_used="unknown",
            error=str(e),
        )


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for a text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for encoding selection

    Returns:
        Estimated token count
    """
    try:
        # Map model names to tiktoken encodings
        if "claude" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "gpt-4" in model.lower() or "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Default to cl100k_base (used by GPT-4, Claude, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except Exception as e:
        logger.warning(f"Error estimating tokens: {str(e)}, using word count")
        # Fallback: rough estimate (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.33)


def parse_openai_response(response: Any, model_name: str) -> ResponseMetrics:
    """
    Parse OpenAI API response to extract metrics.

    Args:
        response: OpenAI API response object
        model_name: Model identifier

    Returns:
        ResponseMetrics with extracted data
    """
    try:
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Calculate cost
        cost = PricingCalculator.calculate_cost(
            model=model_name, input_tokens=input_tokens, output_tokens=output_tokens
        )

        return ResponseMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_used=model_name,
        )

    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
        return ResponseMetrics(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            model_used=model_name,
            error=str(e),
        )


def aggregate_metrics(metrics_list: list[ResponseMetrics]) -> dict[str, Any]:
    """
    Aggregate multiple ResponseMetrics into summary statistics.

    Args:
        metrics_list: List of ResponseMetrics objects

    Returns:
        Dictionary with aggregated statistics
    """
    total_input = sum(m.input_tokens for m in metrics_list)
    total_output = sum(m.output_tokens for m in metrics_list)
    total_cost = sum(m.cost_usd for m in metrics_list if m.cost_usd is not None)
    total_latency = sum(m.latency_seconds for m in metrics_list)

    successful = [m for m in metrics_list if m.error is None]
    failed = [m for m in metrics_list if m.error is not None]

    return {
        "total_requests": len(metrics_list),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_cost_usd": round(total_cost, 6),
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_seconds": round(
            total_latency / len(metrics_list) if metrics_list else 0, 3
        ),
        "avg_cost_per_request": round(
            total_cost / len(metrics_list) if metrics_list else 0, 6
        ),
    }
