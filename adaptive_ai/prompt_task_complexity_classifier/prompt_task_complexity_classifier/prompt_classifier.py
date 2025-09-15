import logging
import traceback
import torch
from functools import lru_cache
from typing import Dict, Any
from transformers import AutoConfig, AutoTokenizer
from .task_complexity_model import CustomModel
from .config import get_config


class PromptClassifier:
    """Prompt task complexity classifier service for ML inference."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.model = None
        self.tokenizer = None
        self.torch = torch

    def load_model(self) -> None:
        """Load the model on startup."""
        config = get_config()
        model_name = config.deployment.model_name

        print(f"🚀 Loading model: {model_name}")
        print(
            f"🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model config and create custom model
        AutoConfig.from_pretrained(model_name)
        self.model = CustomModel(
            target_sizes=config.target_sizes,
            task_type_map=config.task_type_map,
            weights_map=config.weights_map,
            divisor_map=config.divisor_map,
        ).from_pretrained(model_name)

        if torch.cuda.is_available():
            assert self.model is not None  # Type assertion for mypy
            self.model = self.model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU")

        assert self.model is not None  # Type assertion for mypy
        self.model.eval()
        print("✅ Model ready!")

    def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """Classify a single prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        encoded = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU if available
        if self.torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Run inference
        with self.torch.no_grad():
            try:
                result = self.model(encoded)
                # Extract single result from batch
                return {
                    key: values[0] if isinstance(values, list) else values
                    for key, values in result.items()
                }
            except Exception as e:
                # Log the full error with context and stacktrace
                self.logger.error(
                    f"Model inference failed for prompt (length: {len(prompt)}): {str(e)}"
                )
                self.logger.error(f"Full stacktrace: {traceback.format_exc()}")

                # Re-raise the exception to propagate to FastAPI layer for HTTP 5xx
                raise RuntimeError(f"Model inference failed: {str(e)}") from e


@lru_cache(maxsize=1)
def get_prompt_classifier() -> PromptClassifier:
    """Get cached PromptClassifier instance.

    Uses LRU cache to ensure only one instance is created and reused
    across all requests, avoiding expensive model reloading.

    Returns:
        PromptClassifier: Cached classifier instance with loaded model
    """
    classifier = PromptClassifier()
    classifier.load_model()
    return classifier
