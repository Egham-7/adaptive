"""Constants for the NVIDIA prompt classifier service."""

# Model configuration
MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"

# GPU configuration
GPU_TYPE = "T4"

# Performance settings
MAX_SEQUENCE_LENGTH = 512
DEFAULT_CONCURRENCY_LIMIT = 10
DEFAULT_SCALEDOWN_WINDOW = 300
