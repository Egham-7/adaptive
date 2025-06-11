import os
import yaml
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, Any


class Settings(BaseSettings):
    # Server settings
    port: int = 8000
    host: str = "0.0.0.0"

    # LitServe settings
    accelerator: str = "auto"
    devices: str = "auto"
    max_batch_size: int = 8
    batch_timeout: float = 0.05

    # Model settings
    default_model: str = "gpt-3.5-turbo"
    model_selection_threshold: float = 0.7

    # Model configuration file path
    model_config_path: str = "config.yaml"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Environment
    environment: str = "development"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_prefix = "ADAPTIVE_AI_"

    def get_model_config_path(self) -> str:
        """Get the absolute path to the model configuration file"""
        if os.path.isabs(self.model_config_path):
            return self.model_config_path

        # In Docker container, we're in /app directory
        # Check if we're in a Docker container or local development
        if os.path.exists("/app/adaptive_ai"):
            # Docker environment - config should be in /app
            return os.path.join("/app/adaptive_ai", self.model_config_path)
        else:

            # Local development - get project root and look in adaptive_ai subdir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up from core/ to adaptive_ai/ then look for config
            adaptive_ai_root = os.path.dirname(current_dir)
            return os.path.join(adaptive_ai_root, self.model_config_path)

    def load_model_config(self) -> Dict[str, Any]:
        """Load the model configuration YAML file"""
        config_path = self.get_model_config_path()
        return _load_model_config(config_path)

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities from configuration"""
        config = self.load_model_config()
        return config.get("model_capabilities", {})

    def get_task_model_mappings(self) -> Dict[str, Any]:
        """Get task to model mappings from configuration"""
        config = self.load_model_config()
        return config.get("task_model_mappings", {})

    def get_task_parameters(self) -> Dict[str, Any]:
        """Get task parameters from configuration"""
        config = self.load_model_config()
        return config.get("task_parameters", {})


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def _load_model_config(config_path: str) -> Dict[str, Any]:
    """Load the model configuration YAML file (cached)"""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing model configuration YAML: {e}")
