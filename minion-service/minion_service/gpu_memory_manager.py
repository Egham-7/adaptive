# Standard library imports
from typing import Tuple, Callable, Optional

# Optional GPU support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMemoryManager:
    """Manages GPU memory tracking and availability checks."""

    def __init__(self, logger_callback: Optional[Callable] = None):
        self._logger_callback = logger_callback
        self._gpu_device_count = (
            torch.cuda.device_count()
            if TORCH_AVAILABLE and torch.cuda.is_available()
            else 0
        )

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use."""
        return (
            TORCH_AVAILABLE and torch.cuda.is_available() and self._gpu_device_count > 0
        )

    def get_memory_info(self) -> Tuple[float, float, float]:
        """Get GPU memory information in GB: (used, free, total) across all devices."""
        if not self.is_gpu_available():
            return (0.0, 0.0, 0.0)

        total_used_gb = 0.0
        total_free_gb = 0.0
        total_capacity_gb = 0.0

        try:
            # Check all available GPU devices using cached count
            for device_id in range(self._gpu_device_count):
                try:
                    # Use the same method VLLM uses - actual GPU memory usage
                    memory_info = torch.cuda.mem_get_info(device_id)
                    free_bytes, total_bytes = memory_info
                    used_bytes = total_bytes - free_bytes

                    # Convert to GB and accumulate
                    total_used_gb += used_bytes / (1024**3)
                    total_free_gb += free_bytes / (1024**3)
                    total_capacity_gb += total_bytes / (1024**3)

                    self._log(
                        "gpu_device_memory_detail",
                        f"Device {device_id}: {used_bytes/(1024**3):.2f}GB used, {free_bytes/(1024**3):.2f}GB free",
                    )
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

    def has_sufficient_memory(
        self, required_gb: float, reserve_gb: float = 2.0
    ) -> bool:
        """Check if there's sufficient GPU memory available."""
        _, free_gb, _ = self.get_memory_info()
        return free_gb >= required_gb + reserve_gb

    def cleanup_memory(self):
        """Force GPU memory cleanup."""
        if self.is_gpu_available():
            torch.cuda.empty_cache()

    def estimate_model_memory_gb(self, model_name: str) -> float:
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
