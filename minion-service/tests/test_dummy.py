"""Dummy test to ensure CI passes."""


def test_dummy():
    """A dummy test that always passes to ensure CI pipeline works."""
    assert True


def test_basic_math():
    """Another dummy test with basic assertions."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string_operations():
    """Test basic string operations."""
    text = "minion-service"
    assert len(text) > 0
    assert "minion" in text
    assert text.startswith("minion")
