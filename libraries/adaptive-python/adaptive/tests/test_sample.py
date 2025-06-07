def test_always_passes():
    """This test will always pass."""
    assert True


def test_simple_addition():
    """Test that Python can add numbers correctly."""
    assert 1 + 1 == 2


def test_string_operations():
    """Test basic string operations."""
    assert "hello" + " world" == "hello world"
    assert "hello".upper() == "HELLO"
    assert "WORLD".lower() == "world"


class TestClass:
    """A test class with multiple test methods."""

    def test_class_method(self):
        """Test method within a class."""
        x = "this"
        assert "h" in x

    def test_another_method(self):
        """Another test method that will pass."""
        assert 10 > 5
