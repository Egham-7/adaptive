#!/usr/bin/env python3
"""
Test script for model extraction functionality.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import shutil

# Add the script directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from extract_provider_models import ModelExtractor, ModelInfo, ProviderConfig


async def test_model_info_creation():
    """Test ModelInfo dataclass creation."""
    print("🧪 Testing ModelInfo creation...")
    
    model = ModelInfo(
        id="test-model",
        name="Test Model",
        provider="test",
        type="text",
        max_tokens=4096,
        context_length=8192,
        supports_function_calling=True,
        supports_vision=False,
        supports_streaming=True
    )
    
    assert model.id == "test-model"
    assert model.supports_function_calling is True
    print("✅ ModelInfo creation works correctly")


async def test_provider_config():
    """Test provider configuration."""
    print("🧪 Testing provider configurations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        extractor = ModelExtractor(Path(temp_dir))
        
        # Check that all expected providers are configured
        expected_providers = {
            "openai", "anthropic", "google", "groq", 
            "deepseek", "grok"
        }
        
        assert set(extractor.providers.keys()) == expected_providers
        
        # Check OpenAI configuration
        openai_config = extractor.providers["openai"]
        assert openai_config.name == "OpenAI"
        assert "api.openai.com" in openai_config.api_base
        assert openai_config.models_endpoint == "/models"
        
        print("✅ Provider configurations are correct")


async def test_yaml_structure():
    """Test YAML file creation structure."""
    print("🧪 Testing YAML structure...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        extractor = ModelExtractor(Path(temp_dir))
        
        # Create test models
        test_models = {
            "test_provider": [
                ModelInfo(
                    id="model-1",
                    name="Test Model 1",
                    provider="test_provider",
                    context_length=8192,
                    supports_function_calling=True
                ),
                ModelInfo(
                    id="model-2",
                    name="Test Model 2",
                    provider="test_provider",
                    max_tokens=4096,
                    supports_vision=True
                )
            ]
        }
        
        # Override provider config for test
        extractor.providers["test_provider"] = ProviderConfig(
            name="Test Provider",
            api_base="https://test.api",
            models_endpoint="/models",
            headers={}
        )
        
        # Save to YAML
        extractor.save_to_yaml(test_models)
        
        # Check files were created
        provider_dir = Path(temp_dir) / "test_provider"
        yaml_file = provider_dir / "test_provider_models.yaml"
        summary_file = Path(temp_dir) / "extraction_summary.yaml"
        
        assert provider_dir.exists()
        assert yaml_file.exists()
        assert summary_file.exists()
        
        # Check YAML content structure
        import yaml
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        assert "provider" in data
        assert "models" in data
        assert data["provider"]["name"] == "Test Provider"
        assert data["provider"]["total_models"] == 2
        assert len(data["models"]) == 2
        
        print("✅ YAML structure is correct")


async def test_capability_detection():
    """Test model capability detection methods."""
    print("🧪 Testing capability detection...")
    
    extractor = ModelExtractor()
    
    # Test OpenAI function calling detection
    assert extractor._supports_function_calling("gpt-4") is True
    assert extractor._supports_function_calling("gpt-3.5-turbo") is True
    assert extractor._supports_function_calling("text-embedding-ada-002") is False
    
    # Test vision support detection
    assert extractor._supports_vision("gpt-4o") is True
    assert extractor._supports_vision("gpt-4-vision-preview") is True
    assert extractor._supports_vision("gpt-3.5-turbo") is False
    
    # Test context length detection (note: uses substring matching)
    assert extractor._get_context_length("gpt-4") == 8192
    assert extractor._get_context_length("gpt-4-turbo-preview") == 128000  # Contains "gpt-4-turbo"
    assert extractor._get_context_length("unknown-model") is None
    
    print("✅ Capability detection works correctly")


async def test_error_handling():
    """Test error handling for missing API keys."""
    print("🧪 Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        async with ModelExtractor(Path(temp_dir)) as extractor:
            # Test with empty API key (should skip gracefully)
            config = ProviderConfig(
                name="Test Provider",
                api_base="https://nonexistent.api",
                models_endpoint="/models",
                headers={"Authorization": "Bearer "},  # Empty key
                requires_auth=True
            )
            
            models = await extractor._fetch_models("test", config)
            assert models == []  # Should return empty list, not crash
    
    print("✅ Error handling works correctly")


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting model extraction tests...\n")
    
    try:
        await test_model_info_creation()
        await test_provider_config()
        await test_yaml_structure()
        await test_capability_detection()
        await test_error_handling()
        
        print("\n🎉 All tests passed! The extraction script is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)