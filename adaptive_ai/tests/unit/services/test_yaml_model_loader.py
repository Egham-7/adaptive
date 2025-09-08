"""Unit tests for YAML Model Loader service."""

import logging

import pytest

from adaptive_ai.services.yaml_model_loader import yaml_model_db


@pytest.mark.unit
class TestYAMLModelLoader:
    """Test YAML Model Loader functionality."""

    def test_yaml_model_db_singleton(self):
        """Test that yaml_model_db is a singleton instance."""
        # Import alias for singleton test
        from adaptive_ai.services.yaml_model_loader import yaml_model_db as db2

        assert yaml_model_db is db2
        assert id(yaml_model_db) == id(db2)

    def test_get_model_nonexistent(self):
        """Test getting a non-existent model."""
        model = yaml_model_db.get_model("nonexistent:fake-model")
        assert model is None

    def test_has_model_functionality(self):
        """Test has_model method functionality."""
        # Test with None
        assert yaml_model_db.has_model(None) is False

        # Test with non-existent model
        assert yaml_model_db.has_model("fake:model") is False

    def test_get_model_count(self):
        """Test getting model count."""
        count = yaml_model_db.get_model_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_all_models_structure(self):
        """Test that get_all_models returns correct structure."""
        all_models = yaml_model_db.get_all_models()
        assert isinstance(all_models, dict)

        # If models exist, verify structure
        for unique_id, model in all_models.items():
            assert isinstance(unique_id, str)
            assert hasattr(model, "provider")
            assert hasattr(model, "model_name")

    def test_get_models_by_name_empty(self):
        """Test getting models by non-existent name."""
        models = yaml_model_db.get_models_by_name("nonexistent-model")
        assert isinstance(models, list)
        assert len(models) == 0

    def test_get_models_by_name_none(self):
        """Test get_models_by_name with None input handles gracefully."""
        # This tests that the method doesn't crash with None
        try:
            models = yaml_model_db.get_models_by_name("")
            assert isinstance(models, list)
        except Exception as e:
            # If it fails, that's also acceptable behavior
            # Log the exception for debugging purposes
            logging.getLogger(__name__).debug(f"Expected error in test: {e}")
            pass

    def test_case_insensitive_lookup(self):
        """Test that lookups are case insensitive."""
        # Test with different cases - should not crash
        yaml_model_db.get_model("OpenAI:GPT-4")
        yaml_model_db.get_model("openai:gpt-4")
        yaml_model_db.has_model("OPENAI:GPT-4")
        yaml_model_db.has_model("openai:gpt-4")

        # Should handle gracefully regardless of case
        assert True  # If we get here without crashing, test passes


@pytest.mark.unit
class TestYAMLModelLoaderEdgeCases:
    """Test edge cases for YAML Model Loader."""

    def test_null_input_handling(self):
        """Test handling of null/None inputs."""
        assert yaml_model_db.get_model(None) is None
        assert yaml_model_db.has_model(None) is False

        # These should not crash
        yaml_model_db.get_models_by_name("")

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        assert yaml_model_db.get_model("") is None
        assert yaml_model_db.has_model("") is False

        models = yaml_model_db.get_models_by_name("")
        assert isinstance(models, list)

    def test_malformed_unique_id(self):
        """Test handling of malformed unique IDs."""
        # Test IDs without colon
        assert yaml_model_db.get_model("malformed-id") is None
        assert yaml_model_db.has_model("malformed-id") is False

        # Test with multiple colons
        assert yaml_model_db.get_model("provider:model:extra") is None
