"""
Configuration settings for HumanEval benchmarking.

This module provides centralized configuration for benchmark runs,
including model settings, API keys, and execution parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(".env.local")  # Override with local settings


@dataclass
class HumanEvalConfig:
    """Configuration for HumanEval benchmark runs."""

    # Benchmark parameters
    n_samples: int = 200  # Number of samples per task (DeepEval default)
    k_value: int = 10  # Pass@k metric
    temperature: float = 0.8
    max_tokens: int = 1024

    # Execution settings
    parallel: bool = True
    max_concurrent_tasks: int = 20
    use_cache: bool = False

    # Results storage
    results_folder: str = "benchmarks/results"

    # Retry configuration
    retry_max_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_exp_base: float = 2.0
    retry_cap_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "HumanEvalConfig":
        """Create config from environment variables."""
        return cls(
            n_samples=int(os.getenv("HUMANEVAL_N_SAMPLES", "200")),
            k_value=int(os.getenv("HUMANEVAL_K_VALUE", "10")),
            parallel=os.getenv("BENCHMARK_PARALLEL", "true").lower() == "true",
            max_concurrent_tasks=int(os.getenv("MAX_CONCURRENT_TASKS", "20")),
            use_cache=os.getenv("USE_CACHE", "false").lower() == "true",
            results_folder=os.getenv("DEEPEVAL_RESULTS_FOLDER", "benchmarks/results"),
            retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
            retry_initial_seconds=float(os.getenv("RETRY_INITIAL_SECONDS", "1.0")),
            retry_exp_base=float(os.getenv("RETRY_EXP_BASE", "2.0")),
            retry_cap_seconds=float(os.getenv("RETRY_CAP_SECONDS", "10.0")),
        )

    def get_results_path(self, model_name: str) -> Path:
        """Get results directory path for a specific model."""
        safe_name = model_name.replace(":", "_").replace("/", "_")
        return Path(self.results_folder) / safe_name


@dataclass
class ClaudeConfig:
    """Configuration for Claude model."""

    model_name: str = "claude-sonnet-4-5"
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Please set it in .env.local or pass it explicitly."
            )
        return True


@dataclass
class GLMConfig:
    """Configuration for GLM model."""

    model_name: str = "glm-4.6"
    api_key: str | None = None
    api_base: str = "https://open.bigmodel.cn/api/paas/v4/"

    @classmethod
    def from_env(cls) -> "GLMConfig":
        """Create config from environment variables."""
        return cls(
            model_name=os.getenv("GLM_MODEL", "glm-4.6"),
            api_key=os.getenv("GLM_API_KEY"),
            api_base=os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/"),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "GLM_API_KEY not found in environment. "
                "Please set it in .env.local or pass it explicitly."
            )
        return True


@dataclass
class AdaptiveConfig:
    """Configuration for Adaptive routing model."""

    api_key: str | None = None
    api_base: str = "https://api.llmadaptive.uk/v1/"
    models: list[str] = None
    cost_bias: float = 0.5

    def __post_init__(self):
        """Set default models if not provided."""
        if self.models is None:
            self.models = [
                "anthropic:claude-sonnet-4-5-20250929",
                "zai:glm-4.6",
            ]

    @classmethod
    def from_env(cls) -> "AdaptiveConfig":
        """Create config from environment variables."""
        # Parse models list from env (comma-separated)
        models_str = os.getenv("ADAPTIVE_MODELS")
        models = models_str.split(",") if models_str else None

        return cls(
            api_key=os.getenv("ADAPTIVE_API_KEY"),
            api_base=os.getenv("ADAPTIVE_API_BASE", "https://api.llmadaptive.uk/v1/"),
            models=models,
            cost_bias=float(os.getenv("ADAPTIVE_COST_BIAS", "0.5")),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "ADAPTIVE_API_KEY not found in environment. "
                "Please set it in .env.local or pass it explicitly."
            )
        if not self.models or len(self.models) == 0:
            raise ValueError("Adaptive routing requires at least one model")
        return True


class BenchmarkSettings:
    """
    Complete benchmark settings manager.

    This class provides a unified interface to all configuration settings.
    """

    def __init__(self):
        """Initialize settings from environment."""
        self.humaneval = HumanEvalConfig.from_env()
        self.claude = ClaudeConfig.from_env()
        self.glm = GLMConfig.from_env()
        self.adaptive = AdaptiveConfig.from_env()

    def validate_model(self, model_type: str) -> bool:
        """
        Validate configuration for a specific model type.

        Args:
            model_type: One of 'claude', 'glm', 'adaptive'

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if model_type == "claude":
            return self.claude.validate()
        elif model_type == "glm":
            return self.glm.validate()
        elif model_type == "adaptive":
            return self.adaptive.validate()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def print_summary(self):
        """Print configuration summary to console."""
        print("\n" + "=" * 70)
        print("  HumanEval Benchmark Configuration")
        print("=" * 70)
        print("\n  Benchmark Settings:")
        print(f"    Samples per task (n):     {self.humaneval.n_samples}")
        print(f"    Pass@k metric:            pass@{self.humaneval.k_value}")
        print(f"    Temperature:              {self.humaneval.temperature}")
        print(f"    Max tokens:               {self.humaneval.max_tokens}")
        print(f"    Parallel execution:       {self.humaneval.parallel}")
        print(f"    Max concurrent tasks:     {self.humaneval.max_concurrent_tasks}")
        print(f"    Results folder:           {self.humaneval.results_folder}")

        print("\n  Available Models:")
        print(f"    Claude:                   {self.claude.model_name}")
        print(f"    GLM:                      {self.glm.model_name}")
        print(f"    Adaptive:                 {len(self.adaptive.models)} models")
        print(f"      - Cost bias:            {self.adaptive.cost_bias}")
        print(f"      - Models:               {', '.join(self.adaptive.models)}")

        print("\n" + "=" * 70 + "\n")


# Global settings instance
settings = BenchmarkSettings()


# Convenience functions
def get_humaneval_config() -> HumanEvalConfig:
    """Get HumanEval configuration."""
    return settings.humaneval


def get_claude_config() -> ClaudeConfig:
    """Get Claude configuration."""
    return settings.claude


def get_glm_config() -> GLMConfig:
    """Get GLM configuration."""
    return settings.glm


def get_adaptive_config() -> AdaptiveConfig:
    """Get Adaptive configuration."""
    return settings.adaptive
