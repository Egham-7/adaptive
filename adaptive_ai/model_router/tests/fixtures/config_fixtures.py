"""Configuration test fixtures."""

from collections.abc import Generator
from pathlib import Path
import tempfile

import pytest

from model_router.core.config import (
    FastAPIConfig,
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

fastapi:
  workers: 1
  access_log: true
  log_level: "info"

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

fastapi:
  workers: ${TEST_WORKERS:1}
  access_log: ${TEST_ACCESS_LOG:true}
  log_level: "${TEST_FASTAPI_LOG_LEVEL:info}"

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
        fastapi=FastAPIConfig(workers=2, access_log=False, log_level="warning"),
        logging=LoggingConfig(level="WARNING"),
    )


@pytest.fixture
def production_yaml_config() -> Generator[str, None, None]:
    """Create production-like YAML config."""
    yaml_content = """
server:
  host: "${PROD_HOST:0.0.0.0}"
  port: ${PROD_PORT:8000}

fastapi:
  workers: ${PROD_WORKERS:4}
  access_log: ${PROD_ACCESS_LOG:true}
  log_level: "${PROD_FASTAPI_LOG_LEVEL:info}"

logging:
  level: "${PROD_LOG_LEVEL:INFO}"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
