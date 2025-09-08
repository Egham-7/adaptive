"""Configuration test fixtures."""

from collections.abc import Generator
from pathlib import Path
import tempfile

import pytest

from adaptive_ai.core.config import Settings


@pytest.fixture
def temp_yaml_config() -> Generator[str, None, None]:
    """Create a temporary YAML config file."""
    yaml_content = """
server:
  host: "test-host"
  port: 8888

litserve:
  accelerator: "cpu"
  devices: "auto"
  max_batch_size: 16
  batch_timeout: 0.1

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

litserve:
  accelerator: "${TEST_ACCELERATOR:auto}"
  max_batch_size: ${TEST_BATCH_SIZE:8}

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
        server={"host": "test-server", "port": 8999},
        litserve={"max_batch_size": 32, "batch_timeout": 0.01},
        logging={"level": "WARNING"},
    )


@pytest.fixture
def production_yaml_config() -> Generator[str, None, None]:
    """Create production-like YAML config."""
    yaml_content = """
server:
  host: "${PROD_HOST:0.0.0.0}"
  port: ${PROD_PORT:8000}

litserve:
  accelerator: "${PROD_ACCELERATOR:gpu}"
  devices: "${PROD_DEVICES:0,1,2,3}"
  max_batch_size: ${PROD_BATCH_SIZE:64}
  batch_timeout: ${PROD_BATCH_TIMEOUT:0.005}

logging:
  level: "${PROD_LOG_LEVEL:INFO}"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
