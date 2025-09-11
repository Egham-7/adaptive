"""Tests for configuration system with YAML support."""

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from adaptive_ai.core.config import (
    LitServeConfig,
    LoggingConfig,
    ServerConfig,
    Settings,
    get_settings,
)


class TestServerConfig:
    """Test ServerConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "::"
        assert config.port == 8000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(host="127.0.0.1", port=9000)
        assert config.host == "127.0.0.1"
        assert config.port == 9000


class TestLitServeConfig:
    """Test LitServeConfig model."""

    def test_default_values(self):
        """Test default LitServe configuration."""
        config = LitServeConfig()
        assert config.accelerator == "auto"
        assert config.devices == "auto"
        assert config.max_batch_size == 8
        assert config.batch_timeout == 0.05

    def test_custom_values(self):
        """Test custom LitServe configuration."""
        config = LitServeConfig(
            accelerator="gpu", devices="0,1", max_batch_size=16, batch_timeout=0.1
        )
        assert config.accelerator == "gpu"
        assert config.devices == "0,1"
        assert config.max_batch_size == 16
        assert config.batch_timeout == 0.1


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"

    def test_custom_values(self):
        """Test custom logging configuration."""
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        assert settings.server.host == "::"
        assert settings.server.port == 8000
        assert settings.litserve.max_batch_size == 8
        assert settings.logging.level == "INFO"

    def test_env_var_override(self, mock_env_vars):
        """Test environment variable override using nested delimiter format."""
        with mock_env_vars(ADAPTIVE_AI_SERVER__PORT="9999"):
            settings = Settings()
            assert settings.server.port == 9999

    def test_from_yaml_simple(self):
        """Test loading from simple YAML file."""
        yaml_content = """
server:
  host: "0.0.0.0"
  port: 7000
litserve:
  max_batch_size: 16
logging:
  level: "WARNING"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            settings = Settings.from_yaml(temp_path)
            assert settings.server.host == "0.0.0.0"  # noqa: S104
            assert settings.server.port == 7000
            assert settings.litserve.max_batch_size == 16
            assert settings.logging.level == "WARNING"
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_with_env_vars(self):
        """Test YAML loading with environment variable substitution."""
        yaml_content = """
server:
  host: "${TEST_HOST:localhost}"
  port: ${TEST_PORT:8080}
logging:
  level: ${LOG_LEVEL:INFO}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Test with defaults
            settings = Settings.from_yaml(temp_path)
            assert settings.server.host == "localhost"
            assert settings.server.port == 8080
            assert settings.logging.level == "INFO"

            # Test with environment variables
            with patch.dict(
                os.environ,
                {
                    "TEST_HOST": "192.168.1.100",
                    "TEST_PORT": "5555",
                    "LOG_LEVEL": "DEBUG",
                },
                clear=False,
            ):
                settings_env = Settings.from_yaml(temp_path)
                assert settings_env.server.host == "192.168.1.100"
                assert settings_env.server.port == 5555
                assert settings_env.logging.level == "DEBUG"

        finally:
            Path(temp_path).unlink()

    def test_from_yaml_missing_file(self):
        """Test error handling for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml("/nonexistent/file.yaml")

    def test_env_var_substitution_edge_cases(self):
        """Test edge cases in environment variable substitution."""
        # Test with no default value
        yaml_content = """
server:
  host: "${MISSING_VAR}"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            settings = Settings.from_yaml(temp_path)
            # Should keep placeholder when env var doesn't exist
            assert settings.server.host == "${MISSING_VAR}"
        finally:
            Path(temp_path).unlink()

    def test_partial_yaml_config(self):
        """Test YAML config with only partial settings."""
        yaml_content = """
server:
  port: 9090
logging:
  level: "ERROR"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            settings = Settings.from_yaml(temp_path)
            # Should use YAML values where provided
            assert settings.server.port == 9090
            assert settings.logging.level == "ERROR"
            # Should use defaults for missing values
            assert settings.server.host == "::"
            assert settings.litserve.max_batch_size == 8
        finally:
            Path(temp_path).unlink()


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_default(self):
        """Test get_settings without config file."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.server.port == 8000

    def test_get_settings_with_env_config_file(self, mock_env_vars, isolated_cache):
        """Test get_settings with config file from environment."""
        yaml_content = """
server:
  port: 7777
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with mock_env_vars(ADAPTIVE_AI_CONFIG_FILE=temp_path):
                settings = get_settings()
                assert settings.server.port == 7777
        finally:
            Path(temp_path).unlink()

    def test_get_settings_with_explicit_config(self):
        """Test get_settings with explicitly provided config file."""
        yaml_content = """
logging:
  level: "CRITICAL"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            settings = get_settings(config_file=temp_path)
            assert settings.logging.level == "CRITICAL"
        finally:
            Path(temp_path).unlink()

    def test_get_settings_caching(self, isolated_cache):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance due to caching
        assert settings1 is settings2


@pytest.mark.unit
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create comprehensive config
        yaml_content = """
server:
  host: "${APP_HOST:0.0.0.0}"
  port: ${APP_PORT:8888}

litserve:
  accelerator: "${ML_ACCELERATOR:cpu}"
  devices: "${ML_DEVICES:auto}"
  max_batch_size: ${BATCH_SIZE:32}
  batch_timeout: ${BATCH_TIMEOUT:0.02}

logging:
  level: "${LOG_LEVEL:WARNING}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Test with partial environment override
            with patch.dict(
                os.environ,
                {
                    "APP_HOST": "production.example.com",
                    "APP_PORT": "443",
                    "ML_ACCELERATOR": "gpu",
                    "BATCH_SIZE": "64",
                    # LOG_LEVEL not set, should use default from YAML
                },
                clear=False,
            ):
                settings = Settings.from_yaml(temp_path)

                assert settings.server.host == "production.example.com"
                assert settings.server.port == 443
                assert settings.litserve.accelerator == "gpu"
                assert settings.litserve.devices == "auto"  # Uses default
                assert settings.litserve.max_batch_size == 64
                assert settings.litserve.batch_timeout == 0.02  # Uses default
                assert settings.logging.level == "WARNING"  # Uses YAML default

        finally:
            Path(temp_path).unlink()
