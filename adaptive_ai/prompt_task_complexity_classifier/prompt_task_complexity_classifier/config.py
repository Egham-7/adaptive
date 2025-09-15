"""Configuration management for NVIDIA Prompt Classifier service"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseModel):
    """Service configuration settings"""

    name: str = Field(default="nvidia-prompt-classifier", description="Service name")
    modal_url: str = Field(..., description="Modal deployment URL")
    timeout: int = Field(default=120, description="HTTP client timeout in seconds")


class UserTestConfig(BaseModel):
    """Test configuration settings"""

    test_user: str = Field(default="claude_test", description="Test user identifier")
    test_subject: str = Field(default="test_user", description="Test JWT subject")


class DeploymentConfig(BaseModel):
    """Deployment configuration for Modal"""

    app_name: str = Field(
        default="prompt-task-complexity-classifier", description="Modal app name"
    )
    model_name: str = Field(
        default="nvidia/prompt-task-and-complexity-classifier",
        description="HuggingFace model name",
    )
    gpu_type: str = Field(default="T4", description="GPU type for Modal deployment")
    ml_timeout: int = Field(default=600, description="ML container timeout in seconds")
    web_timeout: int = Field(
        default=300, description="Web container timeout in seconds"
    )
    scaledown_window: int = Field(
        default=300, description="Scaledown window in seconds"
    )
    max_containers: int = Field(default=1, description="Maximum number of containers")
    min_containers: int = Field(default=0, description="Minimum number of containers")
    modal_secret_name: str = Field(
        default="jwt", description="Modal secret name for JWT"
    )


class ClassifierConfig(BaseSettings):
    """Main configuration class for the prompt task complexity classifier"""

    service: ServiceConfig = Field(default_factory=lambda: ServiceConfig(modal_url=""))
    test: UserTestConfig = Field(default_factory=lambda: UserTestConfig())
    deployment: DeploymentConfig = Field(default_factory=lambda: DeploymentConfig())

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "ClassifierConfig":
        """Load configuration from YAML file with environment variable substitution

        Args:
            config_path: Path to YAML config file. If None, looks for config.yaml
                        in the project root directory.

        Returns:
            ClassifierConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If required environment variables are missing
        """
        config_file_path: Path
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_file_path = project_root / "config.yaml"
        else:
            config_file_path = Path(config_path)

        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as f:
            yaml_content = f.read()

        # Substitute environment variables
        yaml_content = cls._substitute_env_vars(yaml_content)
        config_data = yaml.safe_load(yaml_content)

        return cls(**config_data)

    @staticmethod
    def _substitute_env_vars(yaml_content: str) -> str:
        """Substitute environment variables in YAML content

        Supports syntax: ${VAR} and ${VAR:default_value}

        Args:
            yaml_content: YAML content as string

        Returns:
            YAML content with environment variables substituted

        Raises:
            ValueError: If required environment variable is missing
        """

        def replace_env_var(match: re.Match[str]) -> str:
            var_expr = match.group(1)

            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
                return os.getenv(var_name, default_value)
            else:
                var_name = var_expr
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(
                        f"Required environment variable '{var_name}' is not set"
                    )
                return value

        # Pattern matches ${VAR} or ${VAR:default}
        pattern = r"\$\{([^}]+)\}"
        return re.sub(pattern, replace_env_var, yaml_content)

    @classmethod
    def from_env(cls) -> "ClassifierConfig":
        """Load configuration from environment variables

        Environment variable format:
        - SERVICE__MODAL_URL=https://example.modal.run
        - AUTH__JWT_SECRET=your_secret_key
        - TEST__TEST_USER=test_user

        Returns:
            ClassifierConfig instance
        """
        # Create with required auth config from environment
        return cls()


def get_config() -> ClassifierConfig:
    """Get the global configuration instance

    Returns:
        ClassifierConfig instance

    Note:
        This function tries to load from config.yaml first, then falls back
        to environment variables if the YAML file doesn't exist.
    """
    config = ClassifierConfig.from_yaml()

    return config
