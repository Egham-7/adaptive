"""Unit tests for configuration management"""

import os
import tempfile
from unittest.mock import patch
import pytest
import yaml

from prompt_task_complexity_classifier.config import (
    ClassifierConfig,
    DeploymentConfig,
    get_config,
)


class TestDeploymentConfig:
    """Test DeploymentConfig model"""

    def test_deployment_config_defaults(self) -> None:
        """Test DeploymentConfig with default values"""
        config = DeploymentConfig()
        assert config.app_name == "prompt-task-complexity-classifier"
        assert config.model_name == "nvidia/prompt-task-and-complexity-classifier"
        assert config.gpu_type == "T4"
        assert config.ml_timeout == 600
        assert config.web_timeout == 300
        assert config.scaledown_window == 300
        assert config.max_containers == 1
        assert config.min_containers == 0
        assert config.modal_secret_name == "jwt"


class TestClassifierConfig:
    """Test ClassifierConfig main configuration class"""

    def test_classifier_config_defaults(self) -> None:
        """Test ClassifierConfig with default values"""
        config = ClassifierConfig()
        assert isinstance(config.deployment, DeploymentConfig)

    @patch.dict(
        os.environ,
        {
            "DEPLOYMENT__APP_NAME": "test-app",
            "DEPLOYMENT__GPU_TYPE": "A100",
        },
    )
    def test_from_env(self) -> None:
        """Test loading configuration from environment variables"""
        config = ClassifierConfig.from_env()
        assert config.deployment.app_name == "test-app"
        assert config.deployment.gpu_type == "A100"

    def test_substitute_env_vars_with_defaults(self) -> None:
        """Test environment variable substitution with default values"""
        yaml_content = """
        deployment:
          app_name: "${APP_NAME:default-app}"
          gpu_type: "${GPU_TYPE:T4}"
        """
        result = ClassifierConfig._substitute_env_vars(yaml_content)
        assert "default-app" in result
        assert "T4" in result

    @patch.dict(os.environ, {"APP_NAME": "env-app"})
    def test_substitute_env_vars_from_env(self) -> None:
        """Test environment variable substitution from actual env vars"""
        yaml_content = """
        deployment:
          app_name: "${APP_NAME:default-app}"
        """
        result = ClassifierConfig._substitute_env_vars(yaml_content)
        assert "env-app" in result
        assert "default-app" not in result

    def test_substitute_env_vars_missing_required(self) -> None:
        """Test that missing required environment variables raise ValueError"""
        yaml_content = """
        deployment:
          app_name: "${MISSING_VAR}"
        """
        with pytest.raises(
            ValueError, match="Required environment variable 'MISSING_VAR' is not set"
        ):
            ClassifierConfig._substitute_env_vars(yaml_content)

    def test_from_yaml_file_not_found(self) -> None:
        """Test from_yaml with non-existent file"""
        with pytest.raises(FileNotFoundError):
            ClassifierConfig.from_yaml("/non/existent/path.yaml")

    def test_from_yaml_success(self) -> None:
        """Test successful YAML loading"""
        yaml_content = {
            "deployment": {
                "app_name": "test-app",
                "model_name": "test/model",
                "gpu_type": "A100",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            config = ClassifierConfig.from_yaml(temp_path)
            assert config.deployment.app_name == "test-app"
            assert config.deployment.model_name == "test/model"
            assert config.deployment.gpu_type == "A100"
            assert config.deployment.app_name == "test-app"
            assert config.deployment.model_name == "test/model"
            assert config.deployment.gpu_type == "A100"
        finally:
            os.unlink(temp_path)


class TestGlobalConfigFunctions:
    """Test global configuration functions"""

    def test_set_and_get_config(self) -> None:
        """Test that get_config returns configuration with expected values"""
        # Get config (should load from YAML or env)
        retrieved_config = get_config()

        # Should have valid deployment config
        assert isinstance(retrieved_config.deployment, DeploymentConfig)
        assert retrieved_config.deployment.app_name  # Should have some value

    def test_get_config_fallback_to_env(self) -> None:
        """Test that get_config falls back to environment when YAML not found"""

        # Mock the from_yaml to raise FileNotFoundError
        with patch.object(ClassifierConfig, "from_yaml", side_effect=FileNotFoundError):
            with patch.object(ClassifierConfig, "from_env") as mock_from_env:
                mock_config = ClassifierConfig()
                mock_from_env.return_value = mock_config

                config = get_config()
                mock_from_env.assert_called_once()
                assert config == mock_config
