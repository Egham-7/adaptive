from typing import Dict, List, Optional
import litgpt  # type: ignore
import threading
import time
from datetime import datetime, timedelta


class ModelManager:
    def __init__(
        self,
        preload_models: Optional[List[str]] = None,
        inactivity_timeout_minutes: int = 30,
    ):
        self.models: Dict[str, litgpt.LLM] = {}
        self.last_used: Dict[str, datetime] = {}
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

    def set_logger_callback(self, callback):
        """Set callback function for logging metrics."""
        self._logger_callback = callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def _preload_models(self, model_names: List[str]):
        """Preload models at startup for better performance."""
        for model_name in model_names:
            try:
                self._log("preloading_model", model_name)
                load_start = time.perf_counter()

                llm = litgpt.LLM.load(model_name)

                load_time = time.perf_counter() - load_start
                self.models[model_name] = llm
                self.last_used[model_name] = datetime.now()

                self._log("model_preloaded", model_name)
                self._log("preload_duration", load_time)
                print(f"Preloaded model: {model_name} in {load_time:.2f}s")

            except Exception as e:
                print(f"Failed to preload model {model_name}: {e}")
                self._log("preload_failed", model_name)

    def get_model(self, model_name: str) -> litgpt.LLM:
        if model_name in self.models:
            self._log("model_cache_hit", 1)
            # Update last used timestamp
            self.last_used[model_name] = datetime.now()
            return self.models[model_name]

        # Model not found - this should only happen if requesting a model not in preloaded list
        self._log("model_not_preloaded", model_name)
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
        )

    def unload_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.last_used:
                del self.last_used[model_name]
            self._log("model_unloaded", model_name)
            self._log("total_models_loaded", len(self.models))
            print(f"Unloaded model: {model_name}")

    def list_loaded_models(self) -> List[str]:
        return list(self.models.keys())

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
                models_to_unload = []

                with self._main_lock:
                    for model_name, last_used_time in self.last_used.items():
                        if current_time - last_used_time > self.inactivity_timeout:
                            models_to_unload.append(model_name)

                for model_name in models_to_unload:
                    self._log("model_auto_unload_triggered", model_name)
                    self.unload_model(model_name)

                # Wait 5 minutes before next cleanup check
                self._shutdown_cleanup.wait(300)

            except Exception as e:
                print(f"Error in cleanup task: {e}")
                self._shutdown_cleanup.wait(60)  # Wait 1 minute on error

    def shutdown(self) -> None:
        """Shutdown the model manager and cleanup resources."""
        self._shutdown_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics about loaded models including last used times."""
        stats = {}
        current_time = datetime.now()

        for model_name in self.models.keys():
            last_used = self.last_used.get(model_name, current_time)
            inactive_duration = current_time - last_used

            stats[model_name] = {
                "last_used": last_used.isoformat(),
                "inactive_minutes": int(inactive_duration.total_seconds() / 60),
                "will_unload_in_minutes": max(
                    0,
                    int(
                        (self.inactivity_timeout - inactive_duration).total_seconds()
                        / 60
                    ),
                ),
            }

        return stats
