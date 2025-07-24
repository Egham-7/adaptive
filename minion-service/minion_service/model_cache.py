# Standard library imports
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

# Third-party imports
from cachetools import LRUCache


class ModelCache:
    """LRU cache for storing and managing loaded models."""

    def __init__(self, max_size: int = 10, logger_callback: Optional[Callable] = None):
        self.models: LRUCache[str, Any] = LRUCache(maxsize=max_size)
        self.last_used: Dict[str, datetime] = {}
        self.model_gpu_memory: Dict[str, float] = {}  # GPU memory usage in GB
        self.model_load_times: Dict[str, float] = {}  # seconds
        self._logger_callback = logger_callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def get(self, model_name: str) -> Optional[Any]:
        """Get a model from cache, updating last used time."""
        if model_name in self.models:
            self.last_used[model_name] = datetime.now()
            self._log("model_cache_hit", 1)
            return self.models[model_name]

        self._log("model_cache_miss", 1)
        return None

    def put(
        self,
        model_name: str,
        model: Any,
        gpu_memory_gb: float = 0.0,
        load_time: float = 0.0,
    ):
        """Store a model in cache with metadata."""
        self.models[model_name] = model
        self.last_used[model_name] = datetime.now()
        self.model_gpu_memory[model_name] = gpu_memory_gb
        self.model_load_times[model_name] = load_time

        self._log("model_cached", model_name)
        self._log("total_models_loaded", len(self.models))

    async def remove(self, model_name: str) -> float:
        """Remove a model from cache and return freed GPU memory."""
        if model_name not in self.models:
            return 0.0

        # Run the deletion in executor to avoid potential blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.models.__delitem__(model_name))

        # Clean up metadata
        if model_name in self.last_used:
            del self.last_used[model_name]
        if model_name in self.model_load_times:
            del self.model_load_times[model_name]

        # Get freed memory before removing from tracking
        freed_memory_gb = self.model_gpu_memory.pop(model_name, 0.0)

        self._log("model_unloaded", model_name)
        self._log("gpu_memory_freed_gb", round(freed_memory_gb, 2))
        self._log("total_models_loaded", len(self.models))

        return freed_memory_gb

    def get_least_recently_used(self) -> Optional[str]:
        """Get the least recently used model for eviction."""
        if not self.models:
            return None

        oldest_time = None
        oldest_model = None

        for model_name in self.models.keys():
            last_used = self.last_used.get(model_name, datetime.min)
            if oldest_time is None or last_used < oldest_time:
                oldest_time = last_used
                oldest_model = model_name

        return oldest_model

    def list_models(self) -> List[str]:
        """Get list of all cached model names."""
        return list(self.models.keys())

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics about loaded models including last used times."""
        current_time = datetime.now()
        return {
            model_name: self._get_model_stat(model_name, current_time)
            for model_name in self.models.keys()
        }

    def _get_model_stat(self, model_name: str, current_time: datetime) -> Dict:
        """Get statistics for a single model."""
        last_used = self.last_used.get(model_name, current_time)
        inactive_duration = current_time - last_used
        return {
            "last_used": last_used.isoformat(),
            "inactive_minutes": int(inactive_duration.total_seconds() / 60),
            "gpu_memory_gb": self.model_gpu_memory.get(model_name, 0.0),
            "load_time_seconds": self.model_load_times.get(model_name, 0.0),
        }

    def get_total_gpu_memory_usage(self) -> float:
        """Get total GPU memory used by all cached models."""
        return sum(self.model_gpu_memory.values())

    def contains(self, model_name: str) -> bool:
        """Check if model is in cache."""
        return model_name in self.models

    def size(self) -> int:
        """Get number of models in cache."""
        return len(self.models)

    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self.models) == 0
