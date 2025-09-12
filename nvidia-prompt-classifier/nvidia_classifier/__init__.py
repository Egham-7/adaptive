"""NVIDIA Prompt Classifier Package.

Clean package containing the NVIDIA prompt classifier model implementation
for Modal deployment and client for API communication.

Modules:
    - nvidia_model: Complete ML model implementation
    - client: Modal API client with authentication and retry logic
"""

from . import nvidia_model

# Conditional import for client to avoid dependency issues in ML containers
try:
    from .client import (
        ModalPromptClassifier,
        get_modal_prompt_classifier,
        ClassificationResult,
    )

    _client_available = True
except ImportError:
    # Client dependencies not available (e.g., in ML container)
    _client_available = False

    # Define placeholder types when imports fail
    class _PlaceholderType:
        pass

    ModalPromptClassifier = _PlaceholderType  # type: ignore
    get_modal_prompt_classifier = None  # type: ignore
    ClassificationResult = _PlaceholderType  # type: ignore

__version__ = "1.0.0"

# Export available components
if _client_available:
    __all__ = [
        "nvidia_model",
        "ModalPromptClassifier",
        "get_modal_prompt_classifier",
        "ClassificationResult",
    ]
else:
    __all__ = ["nvidia_model"]
