"""Configuration test fixtures."""

from collections.abc import Generator
from pathlib import Path
import tempfile

import pytest

from config import (
    LoggingConfig,
    ServerConfig,
    Settings,
)


@pytest.fixture
def temp_yaml_config() -> Generator[str, None, None]:
    """Create a temporary YAML config file."""
    yaml_content = """
server:
  host: "test-host"
  port: 8888

logging:
  level: "DEBUG"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def env_var_yaml_config() -> Generator[str, None, None]:
    """Create YAML config with environment variable substitution."""
    yaml_content = """
server:
  host: "${TEST_HOST:localhost}"
  port: ${TEST_PORT:9000}

logging:
  level: "${TEST_LOG_LEVEL:INFO}"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def minimal_yaml_config() -> Generator[str, None, None]:
    """Create minimal YAML config for testing defaults."""
    yaml_content = """
server:
  port: 7777
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def test_settings() -> Settings:
    """Provide test settings instance."""
    return Settings(
        server=ServerConfig(host="test-server", port=8999),
        logging=LoggingConfig(level="WARNING"),
    )


@pytest.fixture
def production_yaml_config() -> Generator[str, None, None]:
    """Create production-like YAML config."""
    yaml_content = """
server:
  host: "${PROD_HOST:0.0.0.0}"
  port: ${PROD_PORT:8000}

logging:
  level: "${PROD_LOG_LEVEL:INFO}"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
