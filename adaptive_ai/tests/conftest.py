"""Global test configuration and fixtures."""

import os
from unittest.mock import patch

import pytest

from adaptive_ai.core.config import get_settings

# Import fixtures to make them available
from .fixtures.config_fixtures import *  # noqa: F403


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables before each test.

    Removes all environment variables with ADAPTIVE_AI_*, TEST_*, and PROD_* prefixes.
    """
    # Store original env vars
    original_env = dict(os.environ)

    # Clear test-related env vars by prefix
    prefixes = ["ADAPTIVE_AI_", "TEST_", "PROD_"]
    env_keys = list(
        os.environ.keys()
    )  # Create snapshot to avoid mutation during iteration

    for var in env_keys:
        if any(var.startswith(prefix) for prefix in prefixes):
            os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def isolated_cache():
    """Clear LRU caches before test."""
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
