"""NVIDIA Prompt Classifier Package.

Clean package containing the NVIDIA prompt classifier model implementation
for Modal deployment.

Modules:
    - nvidia_model: Complete ML model implementation with GPU acceleration
"""

from . import nvidia_model

__version__ = "1.0.0"

# Export available components
__all__ = ["nvidia_model"]
