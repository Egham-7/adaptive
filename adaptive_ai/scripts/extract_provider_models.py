#!/usr/bin/env python3
"""
Provider Model Extraction Script

This script extracts all available models from supported AI providers and 
organizes them into structured YAML files for easy configuration management.

Usage:
    python extract_provider_models.py [--providers openai,anthropic] [--output-dir models]

Features:
- Async concurrent API calls for better performance
- Comprehensive error handling and retry logic
- Rate limiting to respect API quotas
- Structured YAML output with model metadata
- Support for all major AI providers
- Environment variable configuration
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import aiohttp
import click
import yaml
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Structured model information."""
    id: str
    name: str
    provider: str
    type: Optional[str] = None
    max_tokens: Optional[int] = None
    input_cost_per_1m_tokens: Optional[float] = None
    output_cost_per_1m_tokens: Optional[float] = None
    context_length: Optional[int] = None
    supports_function_calling: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    created: Optional[str] = None
    owned_by: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class ProviderConfig:
    """Provider-specific configuration."""
    name: str
    api_base: str
    models_endpoint: str
    headers: Dict[str, str]
    rate_limit_rpm: int = 60
    requires_auth: bool = True


class ModelExtractor:
    """Main class for extracting models from AI providers."""
    
    def __init__(self, output_dir: Path = Path("models")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.session: Optional[ClientSession] = None
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        # Provider configurations
        self.providers = self._get_provider_configs()
    
    def _get_provider_configs(self) -> Dict[str, ProviderConfig]:
        """Get configuration for all supported providers."""
        return {
            "openai": ProviderConfig(
                name="OpenAI",
                api_base="https://api.openai.com/v1",
                models_endpoint="/models",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                rate_limit_rpm=500
            ),
            "anthropic": ProviderConfig(
                name="Anthropic",
                api_base="https://api.anthropic.com/v1",
                models_endpoint="/models",
                headers={
                    "x-api-key": os.getenv('ANTHROPIC_API_KEY', ''),
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                rate_limit_rpm=100
            ),
            "google": ProviderConfig(
                name="Google AI",
                api_base="https://generativelanguage.googleapis.com/v1beta",
                models_endpoint="/models",
                headers={
                    "Content-Type": "application/json"
                },
                rate_limit_rpm=300
            ),
            "groq": ProviderConfig(
                name="Groq",
                api_base="https://api.groq.com/openai/v1",
                models_endpoint="/models",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                rate_limit_rpm=30
            ),
            "deepseek": ProviderConfig(
                name="DeepSeek",
                api_base="https://api.deepseek.com/v1",
                models_endpoint="/models",
                headers={
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                rate_limit_rpm=100
            ),
            "grok": ProviderConfig(
                name="xAI Grok",
                api_base="https://api.x.ai/v1",
                models_endpoint="/models",
                headers={
                    "Authorization": f"Bearer {os.getenv('XAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                rate_limit_rpm=50
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = ClientTimeout(total=30, connect=10)
        self.session = ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _fetch_models(self, provider_key: str, config: ProviderConfig) -> List[ModelInfo]:
        """Fetch models from a provider's API with retry logic."""
        async with self.semaphore:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            # Skip providers without API keys if required
            if config.requires_auth:
                auth_header = config.headers.get("Authorization", config.headers.get("x-api-key", ""))
                if not auth_header or auth_header.endswith(" "):
                    logger.warning(f"Skipping {config.name}: No API key provided")
                    return []
            
            url = f"{config.api_base}{config.models_endpoint}"
            
            # Special handling for Google AI
            if provider_key == "google":
                api_key = os.getenv('GOOGLE_API_KEY', '')
                if not api_key:
                    logger.warning("Skipping Google AI: No API key provided")
                    return []
                url += f"?key={api_key}"
            
            logger.info(f"Fetching models from {config.name}...")
            
            try:
                async with self.session.get(url, headers=config.headers) as response:
                    if response.status == 401:
                        logger.error(f"{config.name}: Authentication failed")
                        return []
                    elif response.status == 429:
                        logger.warning(f"{config.name}: Rate limited")
                        raise aiohttp.ClientError("Rate limited")
                    elif response.status != 200:
                        logger.error(f"{config.name}: HTTP {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_models(provider_key, config, data)
            
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching models from {config.name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {config.name}: {e}")
                return []
    
    def _parse_models(self, provider_key: str, config: ProviderConfig, data: dict) -> List[ModelInfo]:
        """Parse API response into ModelInfo objects."""
        models = []
        
        try:
            if provider_key == "openai":
                models_data = data.get("data", [])
                for model in models_data:
                    models.append(ModelInfo(
                        id=model["id"],
                        name=model["id"],
                        provider=provider_key,
                        type=model.get("object"),
                        created=model.get("created"),
                        owned_by=model.get("owned_by"),
                        supports_function_calling=self._supports_function_calling(model["id"]),
                        supports_vision=self._supports_vision(model["id"]),
                        supports_streaming=True,
                        context_length=self._get_context_length(model["id"]),
                        max_tokens=self._get_max_tokens(model["id"])
                    ))
            
            elif provider_key == "anthropic":
                models_data = data.get("data", [])
                for model in models_data:
                    models.append(ModelInfo(
                        id=model["id"],
                        name=model.get("display_name", model["id"]),
                        provider=provider_key,
                        type=model.get("type"),
                        created=model.get("created_at"),
                        supports_function_calling=True,
                        supports_vision=self._supports_vision_anthropic(model["id"]),
                        supports_streaming=True,
                        context_length=200000,  # Default for Claude models
                        max_tokens=8192
                    ))
            
            elif provider_key == "google":
                models_data = data.get("models", [])
                for model in models_data:
                    model_id = model["name"].split("/")[-1]  # Extract model ID from full name
                    models.append(ModelInfo(
                        id=model_id,
                        name=model.get("displayName", model_id),
                        provider=provider_key,
                        description=model.get("description"),
                        supports_function_calling=self._check_google_function_calling(model),
                        supports_vision=self._check_google_vision(model),
                        supports_streaming=True,
                        context_length=self._get_google_context_length(model_id),
                        max_tokens=self._get_google_max_tokens(model_id)
                    ))
            
            elif provider_key in ["groq", "deepseek", "grok"]:
                # These providers use OpenAI-compatible format
                models_data = data.get("data", [])
                for model in models_data:
                    models.append(ModelInfo(
                        id=model["id"],
                        name=model["id"],
                        provider=provider_key,
                        type=model.get("object"),
                        created=model.get("created"),
                        owned_by=model.get("owned_by"),
                        supports_function_calling=self._supports_function_calling_provider(provider_key, model["id"]),
                        supports_streaming=True,
                        context_length=self._get_context_length_provider(provider_key, model["id"]),
                        max_tokens=self._get_max_tokens_provider(provider_key, model["id"])
                    ))
            
            logger.info(f"Successfully parsed {len(models)} models from {config.name}")
            
        except Exception as e:
            logger.error(f"Error parsing models from {config.name}: {e}")
        
        return models
    
    def _supports_function_calling(self, model_id: str) -> bool:
        """Check if OpenAI model supports function calling."""
        function_calling_models = {
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        }
        return any(fc_model in model_id for fc_model in function_calling_models)
    
    def _supports_vision(self, model_id: str) -> bool:
        """Check if OpenAI model supports vision."""
        vision_models = {"gpt-4o", "gpt-4-turbo", "gpt-4-vision"}
        return any(vision_model in model_id for vision_model in vision_models)
    
    def _supports_vision_anthropic(self, model_id: str) -> bool:
        """Check if Anthropic model supports vision."""
        return "claude-3" in model_id or "claude-sonnet" in model_id or "claude-opus" in model_id
    
    def _get_context_length(self, model_id: str) -> Optional[int]:
        """Get context length for OpenAI models."""
        context_lengths = [
            # Order matters: more specific patterns first
            ("gpt-4o-mini", 128000),
            ("gpt-4o", 128000),
            ("gpt-4-turbo", 128000),
            ("gpt-3.5-turbo-16k", 16385),
            ("gpt-3.5-turbo", 16385),
            ("gpt-4", 8192),  # Keep this last as it's the most general
        ]
        for model_prefix, length in context_lengths:
            if model_prefix in model_id:
                return length
        return None
    
    def _get_max_tokens(self, model_id: str) -> Optional[int]:
        """Get max output tokens for OpenAI models."""
        max_tokens = [
            # Order matters: more specific patterns first
            ("gpt-4o-mini", 16384),
            ("gpt-4o", 4096),
            ("gpt-4-turbo", 4096),
            ("gpt-3.5-turbo", 4096),
            ("gpt-4", 4096),  # Keep this last as it's the most general
        ]
        for model_prefix, tokens in max_tokens:
            if model_prefix in model_id:
                return tokens
        return None
    
    def _check_google_function_calling(self, model: dict) -> bool:
        """Check if Google model supports function calling."""
        supported_methods = model.get("supportedGenerationMethods", [])
        return "generateContent" in supported_methods
    
    def _check_google_vision(self, model: dict) -> bool:
        """Check if Google model supports vision."""
        input_token_limit = model.get("inputTokenLimit", 0)
        return input_token_limit > 30000  # Vision models typically have higher limits
    
    def _get_google_context_length(self, model_id: str) -> Optional[int]:
        """Get context length for Google models."""
        if "gemini-2.5-flash-lite" in model_id:
            return 128000
        elif "gemini-2.5-flash" in model_id:
            return 1000000
        elif "gemini-2.5-pro" in model_id:
            return 1000000
        return None
    
    def _get_google_max_tokens(self, model_id: str) -> Optional[int]:
        """Get max tokens for Google models."""
        if "lite" in model_id:
            return 4096
        elif "flash" in model_id:
            return 8192
        elif "pro" in model_id:
            return 8192
        return None
    
    def _supports_function_calling_provider(self, provider: str, model_id: str) -> bool:
        """Check function calling support for other providers."""
        provider_support = {
            "groq": {"llama", "mixtral", "gemma"},
            "deepseek": {"deepseek-chat", "deepseek-coder"},
            "grok": {"grok"}
        }
        supported_models = provider_support.get(provider, set())
        return any(supported in model_id for supported in supported_models)
    
    def _get_context_length_provider(self, provider: str, model_id: str) -> Optional[int]:
        """Get context length for provider models."""
        provider_contexts = {
            "groq": 131072,
            "deepseek": 128000,
            "grok": 32768
        }
        return provider_contexts.get(provider)
    
    def _get_max_tokens_provider(self, provider: str, model_id: str) -> Optional[int]:
        """Get max tokens for provider models."""
        provider_max_tokens = {
            "groq": 4096,
            "deepseek": 4096,
            "grok": 4096
        }
        return provider_max_tokens.get(provider)
    
    async def extract_all_models(self, provider_filter: Optional[Set[str]] = None) -> Dict[str, List[ModelInfo]]:
        """Extract models from all providers or filtered subset."""
        providers_to_process = (
            {k: v for k, v in self.providers.items() if k in provider_filter}
            if provider_filter
            else self.providers
        )
        
        tasks = [
            self._fetch_models(provider_key, config)
            for provider_key, config in providers_to_process.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        provider_models = {}
        for (provider_key, _), result in zip(providers_to_process.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract models for {provider_key}: {result}")
                provider_models[provider_key] = []
            else:
                provider_models[provider_key] = result
        
        return provider_models
    
    def save_to_yaml(self, provider_models: Dict[str, List[ModelInfo]]) -> None:
        """Save extracted models to YAML files."""
        timestamp = datetime.now().isoformat()
        
        for provider, models in provider_models.items():
            if not models:
                logger.warning(f"No models found for {provider}, skipping YAML creation")
                continue
            
            # Create provider-specific directory
            provider_dir = self.output_dir / provider
            provider_dir.mkdir(exist_ok=True)
            
            # Prepare YAML data
            yaml_data = {
                "provider": {
                    "name": self.providers[provider].name,
                    "id": provider,
                    "extracted_at": timestamp,
                    "total_models": len(models)
                },
                "models": [
                    {k: v for k, v in asdict(model).items() if v is not None}
                    for model in models
                ]
            }
            
            # Save to YAML file
            yaml_file = provider_dir / f"{provider}_models.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=120
                )
            
            logger.info(f"Saved {len(models)} models for {provider} to {yaml_file}")
        
        # Create summary file
        self._create_summary(provider_models, timestamp)
    
    def _create_summary(self, provider_models: Dict[str, List[ModelInfo]], timestamp: str) -> None:
        """Create a summary YAML file with all providers."""
        summary_data = {
            "extraction_summary": {
                "extracted_at": timestamp,
                "total_providers": len(provider_models),
                "total_models": sum(len(models) for models in provider_models.values())
            },
            "providers": {}
        }
        
        for provider, models in provider_models.items():
            summary_data["providers"][provider] = {
                "name": self.providers[provider].name,
                "total_models": len(models),
                "models": [model.id for model in models[:10]],  # First 10 model IDs
                "has_more": len(models) > 10
            }
        
        summary_file = self.output_dir / "extraction_summary.yaml"
        with open(summary_file, 'w', encoding='utf-8') as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created extraction summary: {summary_file}")


@click.command()
@click.option(
    "--providers",
    default=None,
    help="Comma-separated list of providers to extract (e.g., 'openai,anthropic')"
)
@click.option(
    "--output-dir",
    default="models",
    help="Output directory for YAML files"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging"
)
def main(providers: Optional[str], output_dir: str, verbose: bool) -> None:
    """Extract AI provider models and save to organized YAML files."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    provider_filter = None
    if providers:
        provider_filter = set(p.strip() for p in providers.split(","))
        logger.info(f"Filtering providers: {provider_filter}")
    
    async def run_extraction():
        async with ModelExtractor(Path(output_dir)) as extractor:
            logger.info("Starting model extraction...")
            provider_models = await extractor.extract_all_models(provider_filter)
            
            logger.info("Saving models to YAML files...")
            extractor.save_to_yaml(provider_models)
            
            # Print summary
            total_models = sum(len(models) for models in provider_models.values())
            logger.info(f"✅ Extraction complete! Found {total_models} models across {len(provider_models)} providers")
            
            return provider_models
    
    try:
        asyncio.run(run_extraction())
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()