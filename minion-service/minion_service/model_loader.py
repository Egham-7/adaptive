# Standard library imports
import asyncio
import time
from typing import Optional, Callable, Any, Coroutine

# Third-party imports
from vllm import LLM

# Local imports
from .gpu_memory_manager import GPUMemoryManager


class ModelLoader:
    """Handles the actual loading of models with memory tracking."""

    def __init__(
        self,
        gpu_memory_manager: GPUMemoryManager,
        timeout_seconds: int = 300,
        logger_callback: Optional[Callable] = None,
    ):
        self.gpu_memory_manager = gpu_memory_manager
        self.timeout_seconds = timeout_seconds
        self._logger_callback = logger_callback

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    async def load_model(self, model_name: str) -> tuple[Any, float, float]:
        """
        Load a model and return (model, gpu_memory_gb, load_time).

        Returns:
            tuple: (loaded_model, gpu_memory_used_gb, load_time_seconds)
        """
        self._log("attempting_model_load", model_name)
        load_start = time.perf_counter()

        # Track GPU memory before loading
        pre_used_gb, _, _ = self.gpu_memory_manager.get_memory_info()

        # Load model in executor with timeout
        loop = asyncio.get_event_loop()
        try:

            def load_llm():
                return LLM(
                    model=model_name,
                )

            llm = await asyncio.wait_for(
                loop.run_in_executor(None, load_llm),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Model loading timed out after {self.timeout_seconds}s")

        load_time = time.perf_counter() - load_start

        # Track GPU memory after loading
        post_used_gb, _, _ = self.gpu_memory_manager.get_memory_info()
        model_memory_gb = max(0.0, post_used_gb - pre_used_gb)

        # Use estimation fallback if calculated memory is too low
        if model_memory_gb < 0.1:
            estimated_memory = self.gpu_memory_manager.estimate_model_memory_gb(
                model_name
            )
            self._log(
                "using_estimated_memory",
                f"Calculated {model_memory_gb:.3f}GB, using estimate {estimated_memory:.1f}GB",
            )
            model_memory_gb = estimated_memory

        self._log("model_load_success", model_name)
        self._log("load_duration", load_time)
        self._log("model_gpu_memory_gb", round(model_memory_gb, 2))

        return llm, model_memory_gb, load_time

    async def load_with_retries(
        self,
        model_name: str,
        max_retries: int = 3,
        evict_callback: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
    ) -> tuple[Any, float, float]:
        """
        Load a model with retry logic and optional eviction callback.

        Args:
            model_name: Name of the model to load
            max_retries: Maximum retry attempts
            evict_callback: Function to call to free memory (returns True if eviction occurred)

        Returns:
            tuple: (loaded_model, gpu_memory_used_gb, load_time_seconds)
        """
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self._log("load_attempt", f"{model_name} (attempt {attempt + 1})")
                return await self.load_model(model_name)

            except (RuntimeError, ValueError, ImportError, OSError) as e:
                last_exception = e
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
                        "free memory",
                    ]
                )
                if is_gpu_memory_error:
                    self._log("gpu_oom_detected", f"GPU memory error for {model_name}")

                # If this isn't the last attempt and we have an eviction callback, try freeing space
                if attempt < max_retries and evict_callback and is_gpu_memory_error:
                    if await evict_callback():
                        self._log(
                            "retrying_after_eviction",
                            f"Retrying {model_name} after freeing memory",
                        )
                        continue

            except Exception as e:
                # Catch-all for unexpected errors
                last_exception = e
                error_msg = str(e)
                error_type = type(e).__name__
                self._log(
                    "unexpected_model_load_error",
                    f"{model_name} attempt {attempt + 1} ({error_type}): {error_msg}",
                )

        # All attempts failed
        self._log("model_load_failed_all_attempts", model_name)
        raise ValueError(
            f"Failed to load model '{model_name}' after {max_retries + 1} attempts: {str(last_exception)}"
        )

    def can_load_model(self, model_name: str, reserve_gb: float = 2.0) -> bool:
        """Check if a model can be loaded based on available GPU memory."""
        if not self.gpu_memory_manager.is_gpu_available():
            return True  # If no GPU management, assume it can load

        estimated_memory = self.gpu_memory_manager.estimate_model_memory_gb(model_name)
        return self.gpu_memory_manager.has_sufficient_memory(
            estimated_memory, reserve_gb
        )
