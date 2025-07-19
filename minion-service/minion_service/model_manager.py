from typing import Dict, List, Optional
import litgpt  # type: ignore
import threading
import time
import psutil
from datetime import datetime, timedelta
from cachetools import LRUCache


class ModelManager:
    def __init__(
        self,
        preload_models: Optional[List[str]] = None,
        inactivity_timeout_minutes: int = 30,
        memory_threshold_percent: float = 85.0,
        memory_reserve_gb: float = 2.0,
    ):
        self.memory_threshold_percent = memory_threshold_percent
        self.memory_reserve_gb = memory_reserve_gb

        # Start with a reasonable maxsize, we'll resize dynamically if needed
        self.models: LRUCache[str, litgpt.LLM] = LRUCache(maxsize=10)
        self.last_used: Dict[str, datetime] = {}
        self.failed_models: Dict[str, str] = {}  # Track models that failed to load
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._main_lock = threading.Lock()
        self._logger_callback = None
        self.inactivity_timeout = timedelta(minutes=inactivity_timeout_minutes)
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_cleanup = threading.Event()

        # Preload models at startup
        if preload_models:
            self._preload_models(preload_models)

        # Start background cleanup task
        self._start_cleanup_task()

    def _should_evict_models(self) -> bool:
        """Check if we should evict models due to memory pressure."""
        used_percent, available_gb, has_enough_memory = self._get_memory_info()
        return not has_enough_memory

    def set_logger_callback(self, callback):
        """Set callback function for logging metrics."""
        self._logger_callback = callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def _preload_models(self, model_names: List[str]):
        """Preload models at startup for better performance."""
        failed_models = []
        loaded_models = []

        for model_name in model_names:
            try:
                self._log("preloading_model", model_name)
                load_start = time.perf_counter()

                llm = litgpt.LLM.load(model_name)

                load_time = time.perf_counter() - load_start
                self.models[model_name] = llm
                self.last_used[model_name] = datetime.now()
                loaded_models.append(model_name)

                self._log("model_preloaded", model_name)
                self._log("preload_duration", load_time)
                self._log(
                    "model_preload_success", f"{model_name} loaded in {load_time:.2f}s"
                )

            except Exception as e:
                self._log("preload_failed", model_name)
                self._log("preload_error", f"{model_name}: {str(e)}")
                failed_models.append((model_name, str(e)))
                # Store failed model for tracking
                self.failed_models[model_name] = str(e)

        # Log warnings for failed models
        if failed_models:
            failed_model_names = [name for name, _ in failed_models]
            self._log(
                "model_preload_warnings",
                f"Failed to load {len(failed_models)} model(s)",
            )
            self._log("failed_models", failed_model_names)

            # Log each failure as a warning
            for model_name, error in failed_models:
                self._log("model_load_warning", f"{model_name}: {error}")

        # Only throw fatal error if NO models loaded successfully
        if not loaded_models:
            self._log("fatal_error", "No models loaded - service cannot start")
            self._log("zero_models_loaded", len(model_names))
            raise RuntimeError(
                f"No models loaded successfully from {len(model_names)} attempted models"
            )

        # Log successful startup summary
        self._log(
            "preload_summary",
            {
                "loaded_count": len(loaded_models),
                "failed_count": len(failed_models),
                "total_requested": len(model_names),
                "loaded_models": loaded_models,
                "failed_models": [name for name, _ in failed_models],
            },
        )

        if failed_models:
            self._log(
                "partial_success_warning",
                f"Service starting with {len(loaded_models)}/{len(model_names)} models",
            )

    def get_model(self, model_name: str) -> litgpt.LLM:
        if model_name in self.models:
            self._log("model_cache_hit", 1)
            # Update last used timestamp
            self.last_used[model_name] = datetime.now()
            return self.models[model_name]

        # Model not currently loaded - try to load it with memory management
        return self._load_model_with_memory_management(model_name)

    def _load_model_with_memory_management(self, model_name: str) -> litgpt.LLM:
        """Load a model with automatic memory management and LRU eviction."""
        self._log("model_cache_miss", 1)
        self._log("loading_on_demand", model_name)

        # Check if model previously failed
        if model_name in self.failed_models:
            self._log("model_previously_failed", model_name)
            raise ValueError(
                f"Model '{model_name}' previously failed to load: {self.failed_models[model_name]}"
            )

        with self._main_lock:
            # Double-check if model was loaded by another thread
            if model_name in self.models:
                self.last_used[model_name] = datetime.now()
                return self.models[model_name]

            # Check memory before loading
            used_percent, available_gb, has_enough_memory = self._get_memory_info()
            self._log(
                "pre_load_memory_check",
                {
                    "model": model_name,
                    "memory_used_percent": used_percent,
                    "memory_available_gb": available_gb,
                    "has_enough_memory": has_enough_memory,
                },
            )

            # If not enough memory, reduce cache size to force eviction
            if not has_enough_memory and len(self.models) > 0:
                self._log(
                    "memory_pressure_reducing_cache",
                    f"Memory at {used_percent:.1f}%, reducing cache size to force LRU eviction",
                )

                # Reduce maxsize to current size - 1 to force eviction of LRU item
                new_maxsize = max(1, len(self.models) - 1)
                old_models = dict(self.models)  # Save current models
                self.models = LRUCache(maxsize=new_maxsize)

                # Re-add models (LRUCache will keep only the most recent ones)
                for model_name, model in old_models.items():
                    self.models[model_name] = model

                self._log(
                    "cache_size_reduced",
                    {
                        "old_size": len(old_models),
                        "new_maxsize": new_maxsize,
                        "current_size": len(self.models),
                    },
                )

            # Attempt to load the requested model
            try:
                self._log("attempting_model_load", model_name)
                load_start = time.perf_counter()

                pre_load_memory = psutil.virtual_memory().percent
                llm = litgpt.LLM.load(model_name)
                post_load_memory = psutil.virtual_memory().percent

                load_time = time.perf_counter() - load_start
                memory_increase = post_load_memory - pre_load_memory

                # Successfully loaded
                self.models[model_name] = llm
                self.last_used[model_name] = datetime.now()

                self._log("on_demand_load_success", model_name)
                self._log("load_duration", load_time)
                self._log("memory_increase", memory_increase)
                self._log("total_models_loaded", len(self.models))

                return llm

            except Exception as e:
                error_msg = str(e)
                self._log("on_demand_load_failed", model_name)
                self._log("load_error", f"{model_name}: {error_msg}")

                # Store failure for future reference
                self.failed_models[model_name] = error_msg

                # Check if memory-related error
                if any(
                    keyword in error_msg.lower()
                    for keyword in ["memory", "oom", "cuda out of memory", "allocation"]
                ):
                    self._log(
                        "memory_error_on_demand",
                        "Memory error during on-demand loading",
                    )
                    raise RuntimeError(
                        f"Cannot load model '{model_name}' due to memory constraints: {error_msg}"
                    )

                raise ValueError(f"Failed to load model '{model_name}': {error_msg}")

    def unload_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.last_used:
                del self.last_used[model_name]
            self._log("model_unloaded", model_name)
            self._log("total_models_loaded", len(self.models))

    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())

    def _get_memory_info(self) -> tuple[float, float, bool]:
        """Get current memory information and check if there's enough available."""
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        available_gb = memory.available / (1024**3)
        has_enough_memory = (
            used_percent < self.memory_threshold_percent
            and available_gb > self.memory_reserve_gb
        )
        return used_percent, available_gb, has_enough_memory

    def _start_cleanup_task(self) -> None:
        """Start background task to periodically unload unused models."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_unused_models, daemon=True
            )
            self._cleanup_thread.start()

    def _cleanup_unused_models(self) -> None:
        """Background task that runs periodically to unload unused models."""
        while not self._shutdown_cleanup.is_set():
            try:
                current_time = datetime.now()

                with self._main_lock:
                    models_to_unload = [
                        model_name
                        for model_name, last_used_time in self.last_used.items()
                        if current_time - last_used_time > self.inactivity_timeout
                    ]

                # Log and unload models
                for model_name in models_to_unload:
                    self._log("model_auto_unload_triggered", model_name)
                    self.unload_model(model_name)

                # Wait 5 minutes before next cleanup check
                self._shutdown_cleanup.wait(300)

            except Exception as e:
                self._log("cleanup_error", str(e))
                self._shutdown_cleanup.wait(60)  # Wait 1 minute on error

    def shutdown(self) -> None:
        """Shutdown the model manager and cleanup resources."""
        self._shutdown_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics about loaded models including last used times."""
        current_time = datetime.now()

        return {
            model_name: {
                "last_used": (
                    last_used := self.last_used.get(model_name, current_time)
                ).isoformat(),
                "inactive_minutes": int(
                    (inactive_duration := current_time - last_used).total_seconds() / 60
                ),
                "will_unload_in_minutes": max(
                    0,
                    int(
                        (self.inactivity_timeout - inactive_duration).total_seconds()
                        / 60
                    ),
                ),
            }
            for model_name in self.models.keys()
        }
