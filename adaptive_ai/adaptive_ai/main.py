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

import asyncio
import logging
import sys
import time
from typing import Any

import litserve as ls

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_router import ModelRouter
from adaptive_ai.services.prompt_classifier import get_prompt_classifier

# No custom exceptions needed - using built-in ValueError


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
    It processes individual model selection requests using async Modal-based prompt
    classification and intelligent routing algorithms.

    Request Processing Flow:
    1. decode_request: Parse and validate incoming JSON requests
    2. predict: Async process single request with Modal classification and routing
    3. encode_response: Serialize responses back to JSON

    Key Features:
    - Async processing optimized for Modal integration
    - Modal-based ML prompt classification (task type, complexity)
    - Intelligent model selection with cost optimization
    - Comprehensive logging and metrics
    - Error handling with graceful fallbacks

    Attributes:
        settings: Application configuration from environment
        prompt_classifier: Modal API client for prompt analysis
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

        # Initialize ML components
        self.prompt_classifier = get_prompt_classifier(lit_logger=self)
        self.model_router = ModelRouter(lit_logger=self)

    def decode_request(self, request: ModelSelectionRequest) -> ModelSelectionRequest:
        """Decode and normalize incoming request."""
        # Don't validate here - handle in predict where we can return proper errors
        return request

    async def predict(self, request: ModelSelectionRequest) -> ModelSelectionResponse | dict[str, Any]:
        """Process single model selection request asynchronously."""
        try:
            # Classify prompt using Modal API
            classification_result = await self._classify_prompt_async(request)

            # Process the request with classification results
            response = await self._process_request_async(request, classification_result)

            self.log("predict_completed", {"request_processed": True})
            return response

        except ValueError as e:
            # LitServe limitation: Cannot return custom HTTP status codes
            # Workaround: Include error details in the response body
            error_response: dict[str, Any] = {
                "error": type(e).__name__,
                "message": str(e),
                "provider": None,
                "model": None,
                "alternatives": [],
            }
            self.log("predict_error", {"error": str(e)})
            return error_response

    async def _classify_prompt_async(self, request: ModelSelectionRequest) -> ClassificationResult:
        """Classify single prompt for task type asynchronously."""
        start_time = time.perf_counter()
        
        # Extract prompt from request
        if hasattr(request, 'chat_completion_request') and request.chat_completion_request:
            messages = request.chat_completion_request.messages
            prompt = messages[-1].content if messages else ""
        else:
            prompt = getattr(request, 'prompt', '')
        
        # Use the async Modal classification method directly
        results = await self.prompt_classifier.classify_prompts_async([prompt])
        
        elapsed = time.perf_counter() - start_time
        self.log("modal_classification_time", elapsed)
        
        return results[0] if results else self._get_default_classification()

    def _get_default_classification(self) -> ClassificationResult:
        """Get default classification when Modal fails."""
        return ClassificationResult(
            task_type_1=["Other"],
            task_type_2=["Unknown"],
            task_type_prob=[0.5],
            creativity_scope=[0.5],
            reasoning=[0.5],
            contextual_knowledge=[0.5],
            prompt_complexity_score=[0.5],
            domain_knowledge=[0.5],
            number_of_few_shots=[0],
            no_label_reason=[0.5],
            constraint_ct=[0.5]
        )

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

    async def _process_request_async(
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
        models_input = request.models
        cost_bias = request.cost_bias

        # Select models
        start_time = time.perf_counter()
        task_complexity = (
            classification.prompt_complexity_score[0]
            if classification.prompt_complexity_score
            else 0.5
        )

        # Run model selection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        selected_models = await loop.run_in_executor(
            None,
            self.model_router.select_models,
            task_complexity,
            task_type,
            models_input,
            cost_bias or 0.5,  # Handle None case
        )

        elapsed = time.perf_counter() - start_time
        self.log("model_selection_time", elapsed)

        if not selected_models:
            # This should not happen with proper error handling above, but keep as fallback
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


def create_app() -> ls.LitServer:
    """Create and configure LitServer application."""
    settings = get_settings()

    api = ModelRouterAPI()

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
        workers_per_device=1,  # Single worker for async processing
        timeout=30,  # 30 second timeout for Modal requests
        max_batch_size=1,  # Process one request at a time
        batch_timeout=0.0,  # No batching
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
