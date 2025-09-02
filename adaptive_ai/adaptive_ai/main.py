"""Main LitServe API for intelligent model selection and routing.

This module provides the main entry point for the adaptive_ai service, implementing
a LitServe-based API that handles model selection requests with ML-driven classification
and intelligent routing.

Key Components:
- ModelRouterAPI: Main LitServe API class for handling requests
- ConsoleLogger: Simple console logging for LitServe operations
- Model classification and complexity analysis
- Intelligent model selection with cost optimization

Architecture Flow:
1. Receive ModelSelectionRequest with prompt and preferences
2. Classify prompt using ML models (task type, complexity, etc.)
3. Route to optimal model based on classification and cost bias
4. Return ModelSelectionResponse with selected model and alternatives
"""

import logging
import sys
import time
from typing import Any

import litserve as ls
import tiktoken

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_router import ModelRouter
from adaptive_ai.services.prompt_classifier import get_prompt_classifier


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific loggers
    logger = logging.getLogger(__name__)
    logger.info(
        "Adaptive AI service starting with log level: %s", settings.logging.level
    )

    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class ConsoleLogger(ls.Logger):
    """Simple console logger for LitServe."""

    def process(self, key: str, value: Any) -> None:
        print(f"[LitServe] {key}: {value}", flush=True)


class ModelRouterAPI(ls.LitAPI):
    """LitServe API for intelligent model selection and routing.

    This class implements the main API interface for the adaptive_ai service.
    It processes batches of model selection requests using ML-driven prompt
    classification and intelligent routing algorithms.

    Request Processing Flow:
    1. decode_request: Parse and validate incoming JSON requests
    2. predict: Batch process requests with ML classification and routing
    3. encode_response: Serialize responses back to JSON

    Key Features:
    - Batch processing for optimal GPU utilization
    - ML-based prompt classification (task type, complexity)
    - Intelligent model selection with cost optimization
    - Comprehensive logging and metrics
    - Error handling with graceful fallbacks

    Attributes:
        settings: Application configuration from environment
        tokenizer: Tiktoken tokenizer for token counting
        prompt_classifier: ML model for prompt analysis
        model_router: Intelligent routing engine

    Example Request:
    ```json
    {
        "prompt": "Write a Python function to sort a list",
        "user_id": "user123",
        "cost_bias": 0.7,
        "complexity_threshold": 0.6
    }
    ```

    Example Response:
    ```json
    {
        "provider": "openai",
        "model": "gpt-4",
        "alternatives": [
            {"provider": "anthropic", "model": "claude-3-haiku"}
        ]
    }
    ```
    """

    def setup(self, device: str) -> None:
        """Initialize classifiers and router components.

        Args:
            device: Device to run models on (auto-detected by LitServe)
        """
        self.settings = get_settings()

        # Initialize tokenizer with fallback
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize ML components
        self.prompt_classifier = get_prompt_classifier(lit_logger=self)
        self.model_router = ModelRouter(lit_logger=self)

    def decode_request(self, request: ModelSelectionRequest) -> ModelSelectionRequest:
        """Decode and normalize incoming request."""
        return request

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[ModelSelectionResponse]:
        """Process batch of model selection requests."""
        if not requests:
            return []

        prompts = [req.prompt for req in requests]

        # Run classifications
        classification_results = self._classify_prompts(prompts)

        # Process each request
        responses = []
        for req, classification in zip(requests, classification_results, strict=False):
            response = self._process_request(req, classification)
            responses.append(response)

        self.log("predict_completed", {"batch_size": len(responses)})
        return responses

    def _classify_prompts(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify prompts for task types."""
        start_time = time.perf_counter()
        results = self.prompt_classifier.classify_prompts(prompts)
        elapsed = time.perf_counter() - start_time
        self.log("task_classification_time", elapsed)
        return results

    def _convert_to_task_type(self, task_type_str: str | None) -> TaskType:
        """Convert string task type to TaskType enum."""
        if not task_type_str:
            return TaskType.OTHER

        # Try to find matching TaskType enum value
        for task_type in TaskType:
            if task_type.value == task_type_str:
                return task_type

        # Fallback to OTHER if no match found
        return TaskType.OTHER

    def _process_request(
        self,
        request: ModelSelectionRequest,
        classification: ClassificationResult,
    ) -> ModelSelectionResponse:
        """Process a single model selection request."""
        # Extract parameters
        task_type_str = (
            classification.task_type_1[0] if classification.task_type_1 else None
        )
        task_type = self._convert_to_task_type(task_type_str)
        models_input = self._extract_models_input(request)
        cost_bias = request.cost_bias

        # Select models
        start_time = time.perf_counter()
        task_complexity = (
            classification.prompt_complexity_score[0]
            if classification.prompt_complexity_score
            else 0.5
        )
        selected_models = self.model_router.select_models(
            task_complexity=task_complexity,
            task_type=task_type,
            models_input=models_input,
            cost_bias=cost_bias or 0.5,  # Handle None case
        )
        elapsed = time.perf_counter() - start_time
        self.log("model_selection_time", elapsed)

        if not selected_models:
            raise ValueError("No eligible models found")

        # Build response
        best_model = selected_models[0]

        # Ensure we have valid provider and model names
        if not best_model.provider:
            raise ValueError("Selected model missing provider")
        if not best_model.model_name:
            raise ValueError("Selected model missing model_name")

        alternatives = [
            Alternative(provider=alt.provider, model=alt.model_name)
            for alt in selected_models[1:]
            if alt.provider
            and alt.model_name  # Skip alternatives with missing provider/model
        ]

        return ModelSelectionResponse(
            provider=best_model.provider,
            model=best_model.model_name,
            alternatives=alternatives,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text with fallback.

        Args:
            text: Input text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Fallback approximation: 1 token ≈ 4 characters
                return len(text) // 4
        except Exception as e:
            self.log("tokenizer_fallback", {"error": str(e)})
            return len(text) // 4  # Rough approximation

    def _extract_models_input(
        self, request: ModelSelectionRequest
    ) -> list[ModelCapability] | None:
        """Extract models input from request."""
        return request.models

    def encode_response(self, output: ModelSelectionResponse) -> ModelSelectionResponse:
        """Encode response for transmission."""
        return output


def create_app() -> ls.LitServer:
    """Create and configure LitServer application."""
    settings = get_settings()

    api = ModelRouterAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
        loggers=[ConsoleLogger()],
    )


def main() -> None:
    """Run the LitServe application with proper logging setup."""
    # Setup logging first
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Adaptive AI Service")

    try:
        settings = get_settings()
        app = create_app()

        logger.info(
            "Starting server on %s:%d with %s accelerator",
            settings.server.host,
            settings.server.port,
            settings.litserve.accelerator,
        )

        app.run(
            generate_client_file=False,
            host=settings.server.host,
            port=settings.server.port,
        )

    except Exception as e:
        logger.exception("Failed to start Adaptive AI Service: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
