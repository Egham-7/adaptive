# Standard library imports
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, NamedTuple, Tuple

# Third-party imports
from cachetools import LRUCache
from vllm import LLM

# Optional GPU support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    gpu_memory_reserve_gb: float = 2.0  # Keep this much GPU memory free
    enable_gpu_memory_management: bool = True
    max_load_retries: int = 3  # Maximum retry attempts for model loading
    model_loading_timeout_seconds: int = 300  # 5 minutes timeout for model loading
    cache_max_size: int = 10  # Maximum number of models to keep in cache


class ModelManager:
    def __init__(
        self,
        config: Optional[ModelManagerConfig] = None,
    ):
        self.config = config or ModelManagerConfig()

        # Use configurable cache size
        self.models: LRUCache[str, LLM] = LRUCache(maxsize=self.config.cache_max_size)
        self.last_used: Dict[str, datetime] = {}
        self.load_attempts: Dict[str, ModelLoadAttempt] = {}
        self.model_load_times: Dict[str, float] = {}  # seconds
        self.model_gpu_memory: Dict[str, float] = {}  # GPU memory usage in GB
        self._model_loading_lock = asyncio.Lock()  # Single lock for model loading
        self._logger_callback = None

        # Cache GPU device count for performance
        self._gpu_device_count = (
            torch.cuda.device_count()
            if TORCH_AVAILABLE and torch.cuda.is_available()
            else 0
        )

        # Preloading will be done separately via preload_models_async()

    def set_logger_callback(self, callback):
        """Set callback function for logging metrics."""
        self._logger_callback = callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    async def preload_models_async(self, model_names: List[str]):
        """Preload models at startup with memory-aware parallel loading."""
        failed_models: List[ModelLoadFailure] = []
        loaded_models: List[str] = []

        # Sort models by estimated size (smallest first for better packing)
        sorted_models = sorted(model_names, key=self._estimate_model_memory_gb)

        # Load models with memory-aware concurrency
        loading_lock = asyncio.Lock()  # Protect memory checks and model loading

        async def load_model_with_memory_check(model_name: str):
            async with loading_lock:
                # Check GPU memory before attempting load
                if self.config.enable_gpu_memory_management:
                    _, free_gb, _ = self._get_gpu_memory_info()
                    estimated_memory_gb = self._estimate_model_memory_gb(model_name)

                    if (
                        free_gb
                        < estimated_memory_gb + self.config.gpu_memory_reserve_gb
                    ):
                        self._log(
                            "preload_skipped_memory",
                            f"{model_name}: need {estimated_memory_gb:.1f}GB, only {free_gb:.1f}GB free",
                        )
                        failure = ModelLoadFailure(
                            model_name=model_name,
                            error_message=f"Insufficient GPU memory: need {estimated_memory_gb:.1f}GB, have {free_gb:.1f}GB",
                            timestamp=datetime.now(),
                        )
                        failed_models.append(failure)
                        return

                # Load the model
                try:
                    self._log("preloading_model", model_name)
                    load_start = time.perf_counter()

                    # Track GPU memory before loading
                    pre_used_gb, _, _ = self._get_gpu_memory_info()

                    # Load model in executor with timeout
                    loop = asyncio.get_event_loop()
                    try:
                        llm = await asyncio.wait_for(
                            loop.run_in_executor(None, LLM, model_name),
                            timeout=self.config.model_loading_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"Model loading timed out after {self.config.model_loading_timeout_seconds}s"
                        )

                    load_time = time.perf_counter() - load_start

                    # Track GPU memory after loading
                    post_used_gb, _, _ = self._get_gpu_memory_info()
                    model_memory_gb = max(0.0, post_used_gb - pre_used_gb)

                    # Use estimation fallback if calculated memory is too low
                    if model_memory_gb < 0.1:
                        estimated_memory = self._estimate_model_memory_gb(model_name)
                        self._log(
                            "using_estimated_memory",
                            f"Calculated {model_memory_gb:.3f}GB, using estimate {estimated_memory:.1f}GB",
                        )
                        model_memory_gb = estimated_memory

                    self.models[model_name] = llm
                    self.last_used[model_name] = datetime.now()
                    self.model_gpu_memory[model_name] = model_memory_gb
                    loaded_models.append(model_name)

                    self._log("model_preloaded", model_name)
                    self._log("preload_duration", load_time)
                    self._log("preload_gpu_memory_gb", round(model_memory_gb, 2))

                except Exception as e:
                    self._log("preload_failed", model_name)
                    self._log("preload_error", f"{model_name}: {str(e)}")
                    failure = ModelLoadFailure(
                        model_name=model_name,
                        error_message=str(e),
                        timestamp=datetime.now(),
                    )
                    failed_models.append(failure)
                    self._update_circuit_breaker(model_name, success=False)

        # Process models sequentially but with async I/O for downloads
        for model_name in sorted_models:
            await load_model_with_memory_check(model_name)

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
        # Check if we have sufficient GPU memory before attempting load
        if self.config.enable_gpu_memory_management:
            _, free_gb, _ = self._get_gpu_memory_info()
            estimated_memory_gb = self._estimate_model_memory_gb(model_name)

            if free_gb < estimated_memory_gb + self.config.gpu_memory_reserve_gb:
                self._log(
                    "preload_skipped_memory",
                    f"{model_name}: need {estimated_memory_gb:.1f}GB, only {free_gb:.1f}GB free",
                )
                failure = ModelLoadFailure(
                    model_name=model_name,
                    error_message=f"Insufficient GPU memory: need {estimated_memory_gb:.1f}GB, have {free_gb:.1f}GB",
                    timestamp=datetime.now(),
                )
                failed_models.append(failure)
                return

        try:
            self._log("preloading_model", model_name)
            load_start = time.perf_counter()

            # Track GPU memory before loading
            pre_used_gb, _, _ = self._get_gpu_memory_info()

            llm = LLM(model=model_name)
            load_time = time.perf_counter() - load_start

            # Track GPU memory after loading
            post_used_gb, _, _ = self._get_gpu_memory_info()
            model_memory_gb = post_used_gb - pre_used_gb

            self.models[model_name] = llm
            self.last_used[model_name] = datetime.now()
            self.model_gpu_memory[model_name] = model_memory_gb
            loaded_models.append(model_name)

            self._log("model_preloaded", model_name)
            self._log("preload_duration", load_time)
            self._log("preload_gpu_memory_gb", round(model_memory_gb, 2))
        except Exception as e:
            self._log("preload_failed", model_name)
            self._log("preload_error", f"{model_name}: {str(e)}")
            failure = ModelLoadFailure(
                model_name=model_name, error_message=str(e), timestamp=datetime.now()
            )
            failed_models.append(failure)
            self._update_circuit_breaker(model_name, success=False)

    def _estimate_model_memory_gb(self, model_name: str) -> float:
        """Estimate GPU memory needed for model in GB based on model name."""
        model_lower = model_name.lower()

        # Look for parameter count indicators in model name
        if "7b" in model_lower:
            return 14.0  # ~14GB for 7B parameter models
        elif "8b" in model_lower:
            return 16.0  # ~16GB for 8B parameter models
        elif "3b" in model_lower:
            return 6.0  # ~6GB for 3B parameter models
        elif "1.7b" in model_lower or "1b" in model_lower:
            return 3.0  # ~3GB for smaller models
        elif "14b" in model_lower:
            return 28.0  # ~28GB for 14B parameter models
        elif "70b" in model_lower:
            return 140.0  # ~140GB for 70B parameter models
        else:
            return 8.0  # Default conservative estimate

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

            # Check if we need to free GPU memory before loading
            if self.config.enable_gpu_memory_management:
                await self._ensure_gpu_memory_available()

            return await self._attempt_model_load(model_name)

    async def _attempt_model_load(self, model_name: str) -> LLM:
        """Attempt to load the requested model with LRU eviction on failure."""
        max_retries = self.config.max_load_retries

        for attempt in range(max_retries + 1):
            try:
                self._log(
                    "attempting_model_load", f"{model_name} (attempt {attempt + 1})"
                )
                load_start = time.perf_counter()

                # Track GPU memory before loading
                pre_used_gb, _, _ = self._get_gpu_memory_info()

                # Load model in executor with timeout to avoid hanging
                loop = asyncio.get_event_loop()
                try:
                    llm = await asyncio.wait_for(
                        loop.run_in_executor(None, LLM, model_name),
                        timeout=self.config.model_loading_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"Model loading timed out after {self.config.model_loading_timeout_seconds}s"
                    )

                load_time = time.perf_counter() - load_start

                # Track GPU memory after loading
                post_used_gb, _, _ = self._get_gpu_memory_info()
                # Ensure memory calculation is non-negative (could be negative if other processes freed memory)
                model_memory_gb = max(0.0, post_used_gb - pre_used_gb)

                # If calculated memory is suspiciously low, use estimation as fallback
                if model_memory_gb < 0.1:  # Less than 100MB seems wrong for LLM
                    estimated_memory = self._estimate_model_memory_gb(model_name)
                    self._log(
                        "using_estimated_memory",
                        f"Calculated {model_memory_gb:.3f}GB, using estimate {estimated_memory:.1f}GB",
                    )
                    model_memory_gb = estimated_memory

                self.models[model_name] = llm
                self.last_used[model_name] = datetime.now()
                self.model_load_times[model_name] = load_time
                self.model_gpu_memory[model_name] = model_memory_gb

                self._log("on_demand_load_success", model_name)
                self._log("load_duration", load_time)
                self._log("model_gpu_memory_gb", round(model_memory_gb, 2))
                self._log("total_models_loaded", len(self.models))
                self._log("total_gpu_memory_used", round(post_used_gb, 2))

                # Update circuit breaker on success
                self._update_circuit_breaker(model_name, success=True)

                return llm

            except (RuntimeError, ValueError, ImportError, OSError) as e:
                error_msg = str(e)
                error_type = type(e).__name__
                self._log(
                    "model_load_attempt_failed",
                    f"{model_name} attempt {attempt + 1} ({error_type}): {error_msg}",
                )

                # Check if it's a GPU memory error specifically
                is_gpu_memory_error = any(
                    keyword in error_msg.lower()
                    for keyword in [
                        "cuda out of memory",
                        "gpu memory",
                        "device-side assert",
                    ]
                )
                if is_gpu_memory_error:
                    self._log("gpu_oom_detected", f"GPU memory error for {model_name}")
            except Exception as e:
                # Catch-all for unexpected errors
                error_msg = str(e)
                error_type = type(e).__name__
                self._log(
                    "unexpected_model_load_error",
                    f"{model_name} attempt {attempt + 1} ({error_type}): {error_msg}",
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

    def _get_gpu_memory_info(self) -> Tuple[float, float, float]:
        """Get GPU memory information in GB: (used, free, total) across all devices."""
        if (
            not TORCH_AVAILABLE
            or not torch.cuda.is_available()
            or self._gpu_device_count == 0
        ):
            return (0.0, 0.0, 0.0)

        total_used_gb = 0.0
        total_free_gb = 0.0
        total_capacity_gb = 0.0

        try:
            # Check all available GPU devices using cached count
            for device_id in range(self._gpu_device_count):
                try:
                    # Use the same method VLLM uses - actual GPU memory usage, not just PyTorch allocations
                    memory_info = torch.cuda.mem_get_info(device_id)
                    free_bytes, total_bytes = memory_info
                    used_bytes = total_bytes - free_bytes

                    # Convert to GB and accumulate
                    total_used_gb += used_bytes / (1024**3)
                    total_free_gb += free_bytes / (1024**3)
                    total_capacity_gb += total_bytes / (1024**3)
                    
                    self._log("gpu_device_memory_detail", f"Device {device_id}: {used_bytes/(1024**3):.2f}GB used, {free_bytes/(1024**3):.2f}GB free")
                except RuntimeError as e:
                    # Log but continue if specific device has issues
                    self._log("gpu_device_error", f"Device {device_id}: {str(e)}")
                    continue
        except Exception as e:
            self._log(
                "gpu_memory_info_error", f"Failed to get GPU memory info: {str(e)}"
            )
            return (0.0, 0.0, 0.0)

        return (total_used_gb, total_free_gb, total_capacity_gb)

    async def _ensure_gpu_memory_available(self):
        """Ensure enough GPU memory is available by evicting LRU models if needed."""
        used_gb, free_gb, total_gb = self._get_gpu_memory_info()

        self._log(
            "gpu_memory_check",
            {
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "device_count": self._gpu_device_count,
                "reserve_gb": self.config.gpu_memory_reserve_gb,
            },
        )

        # If we don't have enough free memory, start evicting LRU models
        while free_gb < self.config.gpu_memory_reserve_gb and self.models:
            lru_model = self._get_least_recently_used_model()
            if not lru_model:
                break

            self._log(
                "evicting_for_gpu_memory", f"Unloading {lru_model} to free GPU memory"
            )
            await self.unload_model(lru_model)

            # Re-check memory after unloading
            used_gb, free_gb, total_gb = self._get_gpu_memory_info()
            self._log(
                "gpu_memory_after_eviction",
                {"used_gb": round(used_gb, 2), "free_gb": round(free_gb, 2)},
            )

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

            # Clean up GPU memory tracking and log freed memory
            freed_memory_gb = self.model_gpu_memory.pop(model_name, 0.0)
            self._log("model_unloaded", model_name)
            self._log("gpu_memory_freed_gb", round(freed_memory_gb, 2))
            self._log("total_models_loaded", len(self.models))

            # Force GPU memory cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log current GPU memory status
            used_gb, free_gb, _ = self._get_gpu_memory_info()
            self._log(
                "gpu_memory_after_unload",
                {"used_gb": round(used_gb, 2), "free_gb": round(free_gb, 2)},
            )

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
