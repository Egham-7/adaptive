"""Global test configuration and fixtures."""

import os
from unittest.mock import patch

import pytest

# Import fixtures to make them available
from tests.fixtures.config_fixtures import *  # noqa: F403


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables before each test."""
    # Store original env vars
    original_env = dict(os.environ)

    # Clear test-related env vars
    test_vars = [
        "ADAPTIVE_AI_CONFIG_FILE",
        "ADAPTIVE_AI_SERVER_HOST",
        "ADAPTIVE_AI_SERVER_PORT",
        "ADAPTIVE_AI_LITSERVE_ACCELERATOR",
        "ADAPTIVE_AI_LITSERVE_DEVICES",
        "ADAPTIVE_AI_LITSERVE_MAX_BATCH_SIZE",
        "ADAPTIVE_AI_LITSERVE_BATCH_TIMEOUT",
        "ADAPTIVE_AI_LOGGING_LEVEL",
        "TEST_HOST",
        "TEST_PORT",
        "TEST_ACCELERATOR",
        "TEST_BATCH_SIZE",
        "TEST_LOG_LEVEL",
        "PROD_HOST",
        "PROD_PORT",
        "PROD_ACCELERATOR",
        "PROD_DEVICES",
        "PROD_BATCH_SIZE",
        "PROD_BATCH_TIMEOUT",
        "PROD_LOG_LEVEL",
    ]

    for var in test_vars:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def isolated_cache():
    """Clear LRU caches before test."""
    from adaptive_ai.core.config import get_settings

    # Clear cache before test
    get_settings.cache_clear()

    yield

    # Clear cache after test
    get_settings.cache_clear()


@pytest.fixture
def mock_env_vars():
    """Provide mock environment variables context manager."""

    def _mock_env(**env_vars):
        return patch.dict(os.environ, env_vars, clear=False)

    return _mock_env
