# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, NamedTuple, Callable

# Third-party imports
from vllm import LLM

# Local imports
from .circuit_breaker import CircuitBreaker
from .gpu_memory_manager import GPUMemoryManager
from .model_cache import ModelCache
from .model_loader import ModelLoader


class ModelLoadFailure(NamedTuple):
    model_name: str
    error_message: str
    timestamp: datetime


@dataclass
class ModelManagerConfig:
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    gpu_memory_reserve_gb: float = 2.0
    enable_gpu_memory_management: bool = True
    max_load_retries: int = 3
    model_loading_timeout_seconds: int = 300
    cache_max_size: int = 10


class ModelManager:
    """Orchestrates model loading, caching, and memory management."""

    def __init__(self, config: Optional[ModelManagerConfig] = None):
        self.config = config or ModelManagerConfig()
        self._logger_callback: Optional[Callable] = None
        self._model_loading_lock = asyncio.Lock()

        # Initialize components
        self.gpu_memory_manager = GPUMemoryManager()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            timeout_seconds=self.config.circuit_breaker_timeout_seconds,
        )
        self.model_cache = ModelCache(max_size=self.config.cache_max_size)
        self.model_loader = ModelLoader(
            gpu_memory_manager=self.gpu_memory_manager,
            timeout_seconds=self.config.model_loading_timeout_seconds,
        )

    def set_logger_callback(self, callback: Callable):
        """Set callback function for logging metrics across all components."""
        self._logger_callback = callback
        # Update all components with the new callback
        self.gpu_memory_manager._logger_callback = callback
        self.circuit_breaker._logger_callback = callback
        self.model_cache._logger_callback = callback
        self.model_loader._logger_callback = callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    async def get_model(self, model_name: str) -> LLM:
        """Get a model, loading it if necessary."""
        # Check cache first
        cached_model = self.model_cache.get(model_name)
        if cached_model is not None:
            return cached_model

        # Check circuit breaker before attempting load
        if not self.circuit_breaker.should_attempt_load(model_name):
            raise RuntimeError(
                f"Circuit breaker is open for model '{model_name}' - too many recent failures"
            )

        return await self._load_model_with_management(model_name)

    async def _load_model_with_management(self, model_name: str) -> LLM:
        """Load a model with full memory and concurrency management."""
        async with self._model_loading_lock:
            # Double-check cache after acquiring lock
            cached_model = self.model_cache.get(model_name)
            if cached_model is not None:
                return cached_model

            # Ensure GPU memory is available
            if self.config.enable_gpu_memory_management:
                await self._ensure_gpu_memory_available(model_name)

            # Load the model with retries and eviction callback
            try:
                model, gpu_memory_gb, load_time = (
                    await self.model_loader.load_with_retries(
                        model_name=model_name,
                        max_retries=self.config.max_load_retries,
                        evict_callback=self._evict_lru_model,
                    )
                )

                # Store in cache
                self.model_cache.put(model_name, model, gpu_memory_gb, load_time)

                # Record success in circuit breaker
                self.circuit_breaker.record_success(model_name)

                return model

            except Exception:
                # Record failure in circuit breaker
                self.circuit_breaker.record_failure(model_name)
                raise

    async def _ensure_gpu_memory_available(self, model_name: str):
        """Ensure enough GPU memory is available for the model."""
        estimated_memory = self.gpu_memory_manager.estimate_model_memory_gb(model_name)

        while not self.gpu_memory_manager.has_sufficient_memory(
            estimated_memory, self.config.gpu_memory_reserve_gb
        ):
            if not await self._evict_lru_model():
                # No more models to evict
                used_gb, free_gb, total_gb = self.gpu_memory_manager.get_memory_info()
                raise RuntimeError(
                    f"Insufficient GPU memory for {model_name}: need {estimated_memory:.1f}GB, "
                    f"have {free_gb:.1f}GB free ({used_gb:.1f}GB/{total_gb:.1f}GB used)"
                )

    async def _evict_lru_model(self) -> bool:
        """Evict the least recently used model. Returns True if eviction occurred."""
        lru_model = self.model_cache.get_least_recently_used()
        if not lru_model:
            return False

        freed_memory = await self.model_cache.remove(lru_model)
        self._log("model_evicted_for_memory", f"{lru_model} freed {freed_memory:.1f}GB")
        self.gpu_memory_manager.cleanup_memory()
        return True

    async def preload_models_async(self, model_names: List[str]):
        """Preload models with memory-aware sequential loading."""
        failed_models: List[ModelLoadFailure] = []
        loaded_models: List[str] = []

        # Sort models by estimated size (smallest first for better packing)
        sorted_models = sorted(
            model_names, key=self.gpu_memory_manager.estimate_model_memory_gb
        )

        for model_name in sorted_models:
            try:
                # Check if we can load this model
                if (
                    self.config.enable_gpu_memory_management
                    and not self.model_loader.can_load_model(
                        model_name, self.config.gpu_memory_reserve_gb
                    )
                ):

                    estimated_memory = self.gpu_memory_manager.estimate_model_memory_gb(
                        model_name
                    )
                    _, free_gb, _ = self.gpu_memory_manager.get_memory_info()

                    self._log(
                        "preload_skipped_memory",
                        f"{model_name}: need {estimated_memory:.1f}GB, only {free_gb:.1f}GB free",
                    )
                    failure = ModelLoadFailure(
                        model_name=model_name,
                        error_message=f"Insufficient GPU memory: need {estimated_memory:.1f}GB, have {free_gb:.1f}GB",
                        timestamp=datetime.now(),
                    )
                    failed_models.append(failure)
                    continue

                # Load the model
                self._log("preloading_model", model_name)
                model, gpu_memory_gb, load_time = await self.model_loader.load_model(
                    model_name
                )

                # Store in cache
                self.model_cache.put(model_name, model, gpu_memory_gb, load_time)
                loaded_models.append(model_name)

                self._log("model_preloaded", model_name)
                self._log("preload_gpu_memory_gb", round(gpu_memory_gb, 2))

            except Exception as e:
                self._log("preload_failed", model_name)
                self._log("preload_error", f"{model_name}: {str(e)}")
                failure = ModelLoadFailure(
                    model_name=model_name,
                    error_message=str(e),
                    timestamp=datetime.now(),
                )
                failed_models.append(failure)
                self.circuit_breaker.record_failure(model_name)

        # Log summary
        self._log_preload_summary(loaded_models, failed_models, len(model_names))

        if not loaded_models:
            self._log("fatal_error", "No models loaded - service cannot start")
            raise RuntimeError(
                f"No models loaded successfully from {len(model_names)} attempted models"
            )

    def _log_preload_summary(
        self,
        loaded_models: List[str],
        failed_models: List[ModelLoadFailure],
        total_requested: int,
    ):
        """Log preload summary and warnings."""
        if failed_models:
            self._log(
                "model_preload_warnings",
                f"Failed to load {len(failed_models)} model(s)",
            )
            for failure in failed_models:
                self._log(
                    "model_load_warning",
                    f"{failure.model_name}: {failure.error_message}",
                )
            self._log(
                "partial_success_warning",
                f"Service starting with {len(loaded_models)}/{total_requested} models",
            )

        self._log(
            "preload_summary",
            {
                "loaded_count": len(loaded_models),
                "failed_count": len(failed_models),
                "total_requested": total_requested,
                "loaded_models": loaded_models,
                "failed_models": [f.model_name for f in failed_models],
            },
        )

    async def unload_model(self, model_name: str) -> None:
        """Unload a specific model from cache."""
        freed_memory = await self.model_cache.remove(model_name)
        if freed_memory > 0:
            self.gpu_memory_manager.cleanup_memory()

    def list_loaded_models(self) -> List[str]:
        """Get list of all loaded model names."""
        return self.model_cache.list_models()

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics about loaded models."""
        return self.model_cache.get_stats()

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        used_gb, free_gb, total_gb = self.gpu_memory_manager.get_memory_info()
        return {
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "cached_models_gb": round(self.model_cache.get_total_gpu_memory_usage(), 2),
        }
