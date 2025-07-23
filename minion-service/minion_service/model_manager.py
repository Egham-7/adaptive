from typing import Dict, List, Optional, NamedTuple
from vllm import LLM
import time
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
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60


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

            return await self._attempt_model_load(model_name)

    async def _attempt_model_load(self, model_name: str) -> LLM:
        """Attempt to load the requested model with LRU eviction on failure."""
        max_retries = 3  # Maximum number of models to unload and retry

        for attempt in range(max_retries + 1):
            try:
                self._log(
                    "attempting_model_load", f"{model_name} (attempt {attempt + 1})"
                )
                load_start = time.perf_counter()

                # Load model in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                llm = await loop.run_in_executor(None, LLM, model_name)

                load_time = time.perf_counter() - load_start

                self.models[model_name] = llm
                self.last_used[model_name] = datetime.now()
                self.model_load_times[model_name] = load_time

                self._log("on_demand_load_success", model_name)
                self._log("load_duration", load_time)
                self._log("total_models_loaded", len(self.models))

                # Update circuit breaker on success
                self._update_circuit_breaker(model_name, success=True)

                return llm

            except Exception as e:
                error_msg = str(e)
                self._log(
                    "model_load_attempt_failed",
                    f"{model_name} attempt {attempt + 1}: {error_msg}",
                )

                # If this isn't the last attempt, try freeing space
                if attempt < max_retries and self.models:
                    lru_model = self._get_least_recently_used_model()
                    if lru_model and lru_model != model_name:
                        self._log(
                            "evicting_lru_model",
                            f"Unloading {lru_model} to make space for {model_name}",
                        )
                        await self.unload_model(lru_model)
                        continue

                # Last attempt failed or no models to unload
                self._log("on_demand_load_failed", model_name)
                self._log("load_error", f"{model_name}: {error_msg}")

                # Update circuit breaker on failure
                self._update_circuit_breaker(model_name, success=False)

                raise ValueError(
                    f"Failed to load model '{model_name}' after {attempt + 1} attempts: {error_msg}"
                )

        # This should never be reached due to the raise above, but mypy needs it
        raise RuntimeError("Unexpected code path in _attempt_model_load")

    def _get_least_recently_used_model(self) -> Optional[str]:
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

    async def unload_model(self, model_name: str) -> None:
        if model_name in self.models:
            # Run the deletion in executor to avoid potential blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.models.__delitem__(model_name)
            )
            if model_name in self.last_used:
                del self.last_used[model_name]
            if model_name in self.model_load_times:
                del self.model_load_times[model_name]
            self._log("model_unloaded", model_name)
            self._log("total_models_loaded", len(self.models))

    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())

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
