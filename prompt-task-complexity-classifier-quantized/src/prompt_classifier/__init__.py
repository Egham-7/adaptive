"""
Prompt Task and Complexity Classifier - Quantized

A quantized ONNX version of NVIDIA's prompt task and complexity classifier
optimized for fast CPU inference.

This package provides:
- Quantized ONNX model for efficient inference
- Complete classification pipeline with complexity scoring
- Easy-to-use API compatible with the original model
- Performance optimizations for production use

Example usage:
    >>> from prompt_classifier import QuantizedPromptClassifier
    >>> classifier = QuantizedPromptClassifier.from_pretrained("model_path")
    >>> results = classifier.classify_prompts(["What is machine learning?"])
    >>> print(results[0]["task_type_1"][0])  # "Closed QA"
"""

__version__ = "0.1.0"
__author__ = "Adaptive AI Team"
__email__ = "team@adaptive-ai.com"
__license__ = "Apache-2.0"

from .classifier import QuantizedPromptClassifier
from .utils import load_model_config, validate_model_files

__all__ = [
    "QuantizedPromptClassifier",
    "load_model_config",
    "validate_model_files",
    "__version__",
]
