"""Unit tests for configuration management"""

import os
import tempfile
from unittest.mock import patch
import pytest
import yaml

from prompt_task_complexity_classifier.config import (
    ClassifierConfig,
    ServiceConfig,
    UserTestConfig,
    DeploymentConfig,
    get_config,
)


class TestServiceConfig:
    """Test ServiceConfig model"""

    def test_service_config_defaults(self) -> None:
        """Test ServiceConfig with default values"""
        config = ServiceConfig(modal_url="https://example.com")
        assert config.name == "nvidia-prompt-classifier"
        assert config.modal_url == "https://example.com"
        assert config.timeout == 120

    def test_service_config_custom_values(self) -> None:
        """Test ServiceConfig with custom values"""
        config = ServiceConfig(
            name="custom-service", modal_url="https://custom.com", timeout=60
        )
        assert config.name == "custom-service"
        assert config.modal_url == "https://custom.com"
        assert config.timeout == 60


class TestUserTestConfig:
    """Test UserTestConfig model"""

    def test_test_config_defaults(self) -> None:
        """Test UserTestConfig with default values"""
        config = UserTestConfig()
        assert config.test_user == "claude_test"
        assert config.test_subject == "test_user"

    def test_test_config_custom_values(self) -> None:
        """Test UserTestConfig with custom values"""
        config = UserTestConfig(test_user="custom_user", test_subject="custom_subject")
        assert config.test_user == "custom_user"
        assert config.test_subject == "custom_subject"


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
        assert isinstance(config.service, ServiceConfig)
        assert isinstance(config.test, UserTestConfig)
        assert isinstance(config.deployment, DeploymentConfig)

    @patch.dict(
        os.environ,
        {
            "SERVICE__NAME": "test-service",
            "SERVICE__MODAL_URL": "https://test.com",
        },
    )
    def test_from_env(self) -> None:
        """Test loading configuration from environment variables"""
        config = ClassifierConfig.from_env()
        assert config.service.name == "test-service"
        assert config.service.modal_url == "https://test.com"

    def test_substitute_env_vars_with_defaults(self) -> None:
        """Test environment variable substitution with default values"""
        yaml_content = """
        service:
          name: "${SERVICE_NAME:default-service}"
          modal_url: "${MODAL_URL:https://default.com}"
        """
        result = ClassifierConfig._substitute_env_vars(yaml_content)
        assert "default-service" in result
        assert "https://default.com" in result

    @patch.dict(os.environ, {"SERVICE_NAME": "env-service"})
    def test_substitute_env_vars_from_env(self) -> None:
        """Test environment variable substitution from actual env vars"""
        yaml_content = """
        service:
          name: "${SERVICE_NAME:default-service}"
        """
        result = ClassifierConfig._substitute_env_vars(yaml_content)
        assert "env-service" in result
        assert "default-service" not in result

    def test_substitute_env_vars_missing_required(self) -> None:
        """Test that missing required environment variables raise ValueError"""
        yaml_content = """
        service:
          modal_url: "${MISSING_VAR}"
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
            "service": {
                "name": "test-service",
                "modal_url": "https://test.com",
                "timeout": 60,
            },
            "test": {"test_user": "test_user", "test_subject": "test_subject"},
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
            assert config.service.name == "test-service"
            assert config.service.modal_url == "https://test.com"
            assert config.service.timeout == 60
            assert config.deployment.app_name == "test-app"
            assert config.deployment.model_name == "test/model"
            assert config.deployment.gpu_type == "A100"
        finally:
            os.unlink(temp_path)


class UserTestConfigGlobalFunctions:
    """Test global configuration functions"""

    def test_set_and_get_config(self) -> None:
        """Test setting and getting global configuration"""
        # Create a test config
        test_config = ClassifierConfig()
        test_config.service.name = "test-global-service"

        # Get it back
        retrieved_config = get_config()
        assert retrieved_config.service.name == "test-global-service"

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
