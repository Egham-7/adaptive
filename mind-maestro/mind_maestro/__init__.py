"""Mind Maestro - AI Model Building Framework."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import ModelBuilder
from .training import Trainer
from .evaluation import Evaluator
from .data import DataProcessor

__all__ = [
    "ModelBuilder",
    "Trainer", 
    "Evaluator",
    "DataProcessor",
]