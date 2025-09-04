"""Dummy test to enable pytest coverage collection."""

import pytest


@pytest.mark.unit
def test_basic_functionality() -> None:
    """Basic test that always passes."""
    assert True
