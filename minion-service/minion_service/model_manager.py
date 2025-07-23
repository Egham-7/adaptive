from typing import Dict, List, Optional, NamedTuple, Tuple
from vllm import LLM
import time
import psutil
import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from cachetools import LRUCache


class ModelLoadFailure(NamedTuple):
    model_name: str
    error_message: str
    timestamp: datetime


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ModelLoadAttempt:
    count: int = 0
    last_attempt: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure: Optional[datetime] = None


@dataclass
class ModelManagerConfig:
    memory_reserve_gb: float = 2.0  # Always keep this much memory free
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    model_priority_weights: Dict[str, float] = field(default_factory=dict)
    memory_estimation_factor: float = 1.2  # Safety factor for memory estimation


class ModelManager:
    def __init__(
        self,
        preload_models: Optional[List[str]] = None,
        config: Optional[ModelManagerConfig] = None,
    ):
        self.config = config or ModelManagerConfig()

        # Start with a reasonable maxsize, we'll resize dynamically if needed
        self.models: LRUCache[str, LLM] = LRUCache(maxsize=10)
        self.last_used: Dict[str, datetime] = {}
        self.load_attempts: Dict[str, ModelLoadAttempt] = {}
        self.model_memory_usage: Dict[str, float] = {}  # GB
        self.model_load_times: Dict[str, float] = {}  # seconds
        self._model_loading_lock = asyncio.Lock()  # Single lock for model loading
        self._logger_callback = None

        # Preload models at startup
        if preload_models:
            self._preload_models(preload_models)

    def set_logger_callback(self, callback):
        """Set callback function for logging metrics."""
        self._logger_callback = callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def _preload_models(self, model_names: List[str]):
        """Preload models at startup for better performance."""
        failed_models: List[ModelLoadFailure] = []
        loaded_models: List[str] = []

        # Load models synchronously to avoid event loop conflicts
        for model_name in model_names:
            self._load_single_model(model_name, loaded_models, failed_models)

        if not loaded_models:
            self._log("fatal_error", "No models loaded - service cannot start")
            raise RuntimeError(
                f"No models loaded successfully from {len(model_names)} attempted models"
            )

        self._log_preload_summary(loaded_models, failed_models, len(model_names))

    def _load_single_model(
        self,
        model_name: str,
        loaded_models: List[str],
        failed_models: List[ModelLoadFailure],
    ):
        """Load a single model and update tracking lists."""
        try:
            self._log("preloading_model", model_name)
            load_start = time.perf_counter()

            llm = LLM(model=model_name)
            load_time = time.perf_counter() - load_start

            self.models[model_name] = llm
            self.last_used[model_name] = datetime.now()
            loaded_models.append(model_name)

            self._log("model_preloaded", model_name)
            self._log("preload_duration", load_time)
        except Exception as e:
            self._log("preload_failed", model_name)
            self._log("preload_error", f"{model_name}: {str(e)}")
            failure = ModelLoadFailure(
                model_name=model_name, error_message=str(e), timestamp=datetime.now()
            )
            failed_models.append(failure)
            self._update_circuit_breaker(model_name, success=False)

    def _update_circuit_breaker(self, model_name: str, success: bool):
        """Update circuit breaker state for model loading."""
        if model_name not in self.load_attempts:
            self.load_attempts[model_name] = ModelLoadAttempt()

        attempt = self.load_attempts[model_name]
        attempt.count += 1
        attempt.last_attempt = datetime.now()

        if success:
            attempt.consecutive_failures = 0
            if attempt.circuit_state == CircuitState.HALF_OPEN:
                attempt.circuit_state = CircuitState.CLOSED
                self._log("circuit_breaker_closed", model_name)
        else:
            attempt.consecutive_failures += 1
            attempt.last_failure = datetime.now()

            if (
                attempt.consecutive_failures
                >= self.config.circuit_breaker_failure_threshold
                and attempt.circuit_state == CircuitState.CLOSED
            ):
                attempt.circuit_state = CircuitState.OPEN
                self._log("circuit_breaker_opened", model_name)

    def _should_attempt_load(self, model_name: str) -> bool:
        """Check if we should attempt to load a model based on circuit breaker."""
        if model_name not in self.load_attempts:
            return True

        attempt = self.load_attempts[model_name]

        if attempt.circuit_state == CircuitState.CLOSED:
            return True
        elif attempt.circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            if (
                attempt.last_failure
                and (datetime.now() - attempt.last_failure).total_seconds()
                > self.config.circuit_breaker_timeout_seconds
            ):
                attempt.circuit_state = CircuitState.HALF_OPEN
                self._log("circuit_breaker_half_open", model_name)
                return True
            return False
        elif attempt.circuit_state == CircuitState.HALF_OPEN:
            return True

        return False

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

    async def get_model(self, model_name: str) -> LLM:
        if model_name in self.models:
            self._log("model_cache_hit", 1)
            self.last_used[model_name] = datetime.now()
            return self.models[model_name]

        # Check circuit breaker before attempting load
        if not self._should_attempt_load(model_name):
            raise RuntimeError(
                f"Circuit breaker is open for model '{model_name}' - too many recent failures"
            )

        return await self._load_model_with_memory_management(model_name)

    async def _load_model_with_memory_management(self, model_name: str) -> LLM:
        """Load a model with automatic memory management and LRU eviction."""
        self._log("model_cache_miss", 1)
        self._log("loading_on_demand", model_name)

        async with self._model_loading_lock:
            if model_name in self.models:
                self.last_used[model_name] = datetime.now()
                return self.models[model_name]

            await self._ensure_memory_available(model_name)
            return await self._attempt_model_load(model_name)

    async def _ensure_memory_available(self, model_name: str):
        """Ensure enough memory is available for model loading."""
        estimated_memory = self._estimate_model_memory(model_name)
        used_percent, available_gb = self._get_memory_info()

        self._log(
            "pre_load_memory_check",
            {
                "model": model_name,
                "estimated_memory_gb": estimated_memory,
                "memory_used_percent": used_percent,
                "memory_available_gb": available_gb,
            },
        )

        # Use sophisticated memory management instead of simple LRU eviction
        if available_gb < estimated_memory + self.config.memory_reserve_gb:
            await self._free_memory_for_model(estimated_memory)

    async def _attempt_model_load(self, model_name: str) -> LLM:
        """Attempt to load the requested model."""
        try:
            self._log("attempting_model_load", model_name)
            load_start = time.perf_counter()
            pre_load_memory = psutil.virtual_memory().percent

            # Load model in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            llm = await loop.run_in_executor(None, LLM, model_name)

            load_time = time.perf_counter() - load_start
            post_load_memory = psutil.virtual_memory().percent
            memory_increase = post_load_memory - pre_load_memory

            self.models[model_name] = llm
            self.last_used[model_name] = datetime.now()

            # Track memory usage and load time for future predictions
            self.model_memory_usage[model_name] = (
                memory_increase / 100 * psutil.virtual_memory().total / (1024**3)
            )
            self.model_load_times[model_name] = load_time

            self._log("on_demand_load_success", model_name)
            self._log("load_duration", load_time)
            self._log("memory_increase", memory_increase)
            self._log("total_models_loaded", len(self.models))

            # Update circuit breaker on success
            self._update_circuit_breaker(model_name, success=True)

            return llm

        except Exception as e:
            error_msg = str(e)
            self._log("on_demand_load_failed", model_name)
            self._log("load_error", f"{model_name}: {error_msg}")

            # Update circuit breaker on failure
            self._update_circuit_breaker(model_name, success=False)

            if self._is_memory_error(error_msg):
                self._log(
                    "memory_error_on_demand", "Memory error during on-demand loading"
                )
                raise RuntimeError(
                    f"Cannot load model '{model_name}' due to memory constraints: {error_msg}"
                )

            raise ValueError(f"Failed to load model '{model_name}': {error_msg}")

    def _is_memory_error(self, error_msg: str) -> bool:
        """Check if error is memory-related."""
        memory_keywords = ["memory", "oom", "cuda out of memory", "allocation"]
        return any(keyword in error_msg.lower() for keyword in memory_keywords)

    async def unload_model(self, model_name: str) -> None:
        if model_name in self.models:
            # Run the deletion in executor to avoid potential blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.models.__delitem__(model_name)
            )
            if model_name in self.last_used:
                del self.last_used[model_name]
            if model_name in self.model_memory_usage:
                del self.model_memory_usage[model_name]
            if model_name in self.model_load_times:
                del self.model_load_times[model_name]
            self._log("model_unloaded", model_name)
            self._log("total_models_loaded", len(self.models))

    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())

    def _get_memory_info(self) -> Tuple[float, float]:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        available_gb = memory.available / (1024**3)
        return used_percent, available_gb

    def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory needed for model in GB."""
        # Use historical data if available
        if model_name in self.model_memory_usage:
            return (
                self.model_memory_usage[model_name]
                * self.config.memory_estimation_factor
            )

        # Basic estimation based on model name patterns
        model_lower = model_name.lower()
        if "7b" in model_lower or "7-b" in model_lower:
            return 14.0  # ~14GB for 7B parameter models
        elif "3b" in model_lower or "3-b" in model_lower:
            return 6.0  # ~6GB for 3B parameter models
        elif "1b" in model_lower or "1.7b" in model_lower:
            return 3.0  # ~3GB for smaller models
        elif "14b" in model_lower:
            return 28.0  # ~28GB for 14B parameter models
        else:
            return 8.0  # Default estimation

    async def _free_memory_for_model(self, required_memory_gb: float):
        """Free exactly enough memory for the required model."""
        if not self.models:
            return

        # Calculate current memory usage
        current_usage = sum(
            self.model_memory_usage.get(name, 0) for name in self.models.keys()
        )

        self._log("current_memory_usage", f"{current_usage:.1f}GB")

        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)

        # Check if we need to free memory
        if available_gb >= required_memory_gb + self.config.memory_reserve_gb:
            return

        memory_to_free = (
            required_memory_gb + self.config.memory_reserve_gb - available_gb
        )

        # Get models sorted by priority (LRU + priority weights)
        models_by_priority = self._get_models_by_eviction_priority()

        freed_memory = 0.0
        for model_name in models_by_priority:
            if freed_memory >= memory_to_free:
                break
            model_memory = self.model_memory_usage.get(model_name, 0)
            await self.unload_model(model_name)
            freed_memory += model_memory
            self._log(
                "model_evicted_for_memory", f"{model_name} freed {model_memory:.1f}GB"
            )

    def _get_models_by_eviction_priority(self) -> List[str]:
        """Get models sorted by eviction priority (least important first)."""
        current_time = datetime.now()
        model_scores = []

        for model_name in self.models.keys():
            # Base score on last used time (older = higher eviction priority)
            last_used = self.last_used.get(model_name, current_time)
            time_score = (current_time - last_used).total_seconds()

            # Apply priority weights (lower weight = higher eviction priority)
            priority_weight = self.config.model_priority_weights.get(model_name, 1.0)
            final_score = time_score / priority_weight

            model_scores.append((final_score, model_name))

        # Sort by score (highest first = most likely to evict)
        model_scores.sort(reverse=True)
        return [model_name for _, model_name in model_scores]

    def get_model_stats(self) -> Dict[str, Dict]:
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
        }
