"""Prompt Task Complexity Classifier Package.

Clean package containing the prompt task complexity classifier model implementation
for Modal deployment.

Modules:
    - task_complexity_model: Complete ML model implementation with GPU acceleration
    - config: Configuration management
"""

from . import task_complexity_model, config

__version__ = "1.0.0"

# Export available components
__all__ = ["task_complexity_model", "config"]
