"""Tests for ProfileLoader base class."""

import inspect
from abc import ABC


from adaptive_router.loaders.base import ProfileLoader


class TestProfileLoader:
    """Test ProfileLoader abstract base class."""

    def test_is_abstract_class(self):
        """Test that ProfileLoader is an abstract base class."""
        assert issubclass(ProfileLoader, ABC)
        assert inspect.isabstract(ProfileLoader)

    def test_has_abstract_methods(self):
        """Test that ProfileLoader has abstract methods."""
        abstract_methods = ProfileLoader.__abstractmethods__
        assert "load_profile" in abstract_methods

    def test_health_check_default_implementation(self):
        """Test that health_check returns True by default."""

        # Create a minimal concrete implementation for testing
        class ConcreteLoader(ProfileLoader):
            def load_profile(self):
                raise NotImplementedError("Test implementation")

        loader = ConcreteLoader()
        assert loader.health_check() is True
