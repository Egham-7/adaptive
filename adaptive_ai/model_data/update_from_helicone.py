#!/usr/bin/env python3
"""
Helicone Model Database Updater

Clean, simple script to fetch and update model data from Helicone's API.
Keeps the database fresh with latest pricing and model information.

Usage:
    python update_from_helicone.py                    # Update all providers
    python update_from_helicone.py --providers openai,anthropic  # Specific providers
    python update_from_helicone.py --dry-run          # Show what would be updated
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HeliconeUpdater:
    """Clean updater for Helicone model database."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize updater with data directory."""
        self.data_dir = data_dir or Path(__file__).parent / "data" / "provider_models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Helicone API configuration
        self.base_url = "https://helicone.ai/api/llm-costs"
        self.session = requests.Session()
        self.session.timeout = 30
        self.session.headers.update({
            "User-Agent": "adaptive-ai-helicone-updater/1.0",
            "Accept": "application/json"
        })
        
        logger.info(f"📂 Data directory: {self.data_dir}")
    
    def fetch_provider_models(self, provider: str) -> Dict[str, Any]:
        """Fetch models for a single provider from Helicone API."""
        url = f"{self.base_url}?provider={provider.lower()}"
        
        try:
            logger.info(f"📥 Fetching {provider.upper()}...")
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            models_data = data.get("data", [])
            logger.info(f"  ✅ {provider.upper()}: {len(models_data)} models")
            return models_data
            
        except requests.RequestException as e:
            logger.error(f"  ❌ {provider.upper()}: Failed to fetch - {e}")
            return []
    
    def normalize_model_key(self, model_name: str) -> str:
        """Normalize model name to YAML key format."""
        return (model_name.lower()
                          .replace("/", "_")
                          .replace("-", "_")
                          .replace(".", "_")
                          .replace(" ", "_")
                          .replace(":", "_"))
    
    def convert_to_structured_format(self, provider: str, raw_data: List[Dict]) -> Dict[str, Any]:
        """Convert raw Helicone data to structured YAML format."""
        models = {}
        
        for model in raw_data:
            model_name = model.get("model", "")
            if not model_name:
                continue
            
            # Create normalized key
            model_key = self.normalize_model_key(model_name)
            
            # Build structured model entry
            models[model_key] = {
                "description": "",  # Empty for manual/AI enrichment
                "provider": provider.upper(),
                "model_name": model_name,
                "cost_per_1m_input_tokens": model.get("input_cost_per_1m", 0),
                "cost_per_1m_output_tokens": model.get("output_cost_per_1m", 0),
                "max_context_tokens": None,  # Empty for manual/AI enrichment
                "max_output_tokens": None,   # Empty for manual/AI enrichment
                "supports_streaming": True,  # Default assumption
                "supports_function_calling": None,  # Empty for manual/AI enrichment
                "supports_vision": None,     # Empty for manual/AI enrichment
                "multimodal_capabilities": ["text"],  # Default
                "languages_supported": [],   # Empty for manual/AI enrichment
                "model_size_params": "",     # Empty for manual/AI enrichment
                "latency_tier": "",          # Empty for manual/AI enrichment
                "task_type": "",             # Empty for manual/AI enrichment
                "complexity": "",            # Empty for manual/AI enrichment
                "release_date": None,        # Empty for manual/AI enrichment
                "training_data_cutoff": None,  # Empty for manual/AI enrichment
                "metadata": {
                    "source": "helicone_api",
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "matching_operator": model.get("operator", "equals"),
                    "available_in_playground": model.get("show_in_playground", True)
                }
            }
            
            # Add additional pricing if available (not common in current API format)
            additional_pricing = {}
            if model.get("cache_read_cost_per_1m"):
                additional_pricing["cache_read_per_1m"] = model["cache_read_cost_per_1m"]
            if model.get("cache_write_cost_per_1m"):
                additional_pricing["cache_write_per_1m"] = model["cache_write_cost_per_1m"]
            if model.get("audio_cost_per_second"):
                additional_pricing["audio_per_second"] = model["audio_cost_per_second"]
            
            if additional_pricing:
                models[model_key]["additional_pricing"] = additional_pricing
        
        return models
    
    def create_provider_yaml(self, provider: str, models: Dict[str, Any]) -> Path:
        """Create/update YAML file for provider."""
        yaml_file = self.data_dir / f"{provider.lower()}_models_structured.yaml"
        
        # Create provider data structure
        provider_data = {
            "provider_info": {
                "name": provider.upper(),
                "total_models": len(models),
                "data_source": f"https://helicone.ai/api/llm-costs?provider={provider.lower()}",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "currency": "USD",
                "pricing_unit": "per 1 million tokens"
            },
            "models": models
        }
        
        # Preserve existing enrichments if file exists
        if yaml_file.exists():
            provider_data = self.merge_with_existing(yaml_file, provider_data)
        
        # Write YAML file
        with open(yaml_file, 'w', encoding='utf-8') as f:
            # Header comment
            f.write("# " + "=" * 80 + "\n")
            f.write(f"# Provider: {provider.upper()} Models Database\n")
            f.write(f"# Total Models: {len(models)}\n")
            f.write(f"# Source: Helicone API (https://helicone.ai/api/llm-costs)\n")
            f.write(f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write("# " + "=" * 80 + "\n\n")
            
            # YAML content
            yaml.dump(provider_data, f,
                     default_flow_style=False,
                     sort_keys=False,
                     indent=2,
                     width=100,
                     allow_unicode=True)
        
        logger.info(f"✅ Created {yaml_file.name}")
        return yaml_file
    
    def merge_with_existing(self, yaml_file: Path, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new data with existing enrichments to preserve manual work."""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f)
            
            existing_models = existing_data.get("models", {})
            new_models = new_data["models"]
            
            # Preserve enrichments for existing models
            preserved_count = 0
            for model_key, new_model in new_models.items():
                if model_key in existing_models:
                    existing_model = existing_models[model_key]
                    
                    # Preserve enriched fields if they exist and are not empty
                    enrichment_fields = [
                        "description", "max_context_tokens", "max_output_tokens",
                        "supports_function_calling", "supports_vision", 
                        "languages_supported", "model_size_params", "latency_tier",
                        "task_type", "complexity", "release_date", "training_data_cutoff"
                    ]
                    
                    for field in enrichment_fields:
                        existing_value = existing_model.get(field)
                        if existing_value and existing_value != "" and existing_value != []:
                            new_model[field] = existing_value
                            preserved_count += 1
            
            if preserved_count > 0:
                logger.info(f"  📝 Preserved {preserved_count} enriched fields")
            
            return new_data
            
        except Exception as e:
            logger.warning(f"  ⚠️  Could not merge with existing file: {e}")
            return new_data
    
    def update_master_index(self, updated_providers: List[str]) -> None:
        """Update the master index file with current statistics."""
        index_file = self.data_dir / "00_master_index.yaml"
        
        # Collect provider statistics
        providers_info = {}
        total_models = 0
        
        for yaml_file in self.data_dir.glob("*_models_structured.yaml"):
            if yaml_file.name == "00_master_index.yaml":
                continue
                
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                provider_info = data.get("provider_info", {})
                models = data.get("models", {})
                provider_name = provider_info.get("name", yaml_file.stem.split("_")[0]).lower()
                
                providers_info[provider_name] = {
                    "total_models": len(models),
                    "yaml_file": yaml_file.name,
                    "api_endpoint": provider_info.get("data_source", ""),
                    "sample_models": list(models.keys())[:3] + [f"... and {len(models)-3} more"] if len(models) > 3 else list(models.keys())
                }
                
                total_models += len(models)
                
            except Exception as e:
                logger.warning(f"Could not process {yaml_file}: {e}")
        
        # Create master index
        master_data = {
            "database_info": {
                "title": "Helicone AI Models - Complete Database",
                "description": "Live pricing data for all AI models across major providers",
                "total_providers": len(providers_info),
                "total_models": total_models,
                "data_source": "https://helicone.ai/api/llm-costs",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "currency": "USD",
                "pricing_unit": "per 1 million tokens"
            },
            "providers": dict(sorted(providers_info.items()))
        }
        
        # Write master index
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("# Helicone AI Models - Master Index\n")
            f.write("# " + "=" * 63 + "\n")
            f.write("# \n")
            f.write("# This file provides an overview of all available AI model data\n")
            f.write("# Each provider has its own structured YAML file with complete pricing\n")
            f.write("# \n")
            f.write(f"# 📊 Database Statistics:\n")
            f.write(f"#   • Total Providers: {len(providers_info)}\n")
            f.write(f"#   • Total Models: {total_models}\n")
            f.write(f"#   • Data Source: Live Helicone API\n")
            f.write(f"#   • Last Updated: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("# \n")
            f.write("# 💡 Usage:\n")
            f.write("#   1. Check this file for provider overview\n")
            f.write("#   2. Load specific provider YAML files as needed\n")
            f.write("#   3. All pricing is in USD per 1 million tokens\n")
            f.write("# " + "=" * 63 + "\n\n")
            
            yaml.dump(master_data, f,
                     default_flow_style=False,
                     sort_keys=False,
                     indent=2,
                     width=100,
                     allow_unicode=True)
        
        logger.info(f"✅ Updated master index with {len(providers_info)} providers")
    
    def get_known_providers(self) -> List[str]:
        """Get list of known providers from existing YAML files."""
        providers = []
        for yaml_file in self.data_dir.glob("*_models_structured.yaml"):
            if yaml_file.name == "00_master_index.yaml":
                continue
            provider_name = yaml_file.stem.replace("_models_structured", "")
            providers.append(provider_name)
        return sorted(providers)
    
    def update_providers(self, providers: Optional[List[str]] = None, dry_run: bool = False) -> None:
        """Update specified providers or all known providers."""
        if not providers:
            providers = self.get_known_providers()
            if not providers:
                # Default providers if none exist
                providers = [
                    "openai", "anthropic", "google", "groq", "mistral", "cohere",
                    "deepseek", "together", "fireworks", "novita", "openrouter"
                ]
        
        logger.info(f"🚀 Starting Helicone database update...")
        logger.info(f"📋 Updating {len(providers)} providers: {', '.join(providers)}")
        
        if dry_run:
            logger.info("🧪 DRY RUN MODE - No files will be modified")
            return
        
        updated_providers = []
        failed_providers = []
        
        for provider in providers:
            try:
                # Fetch data from Helicone
                raw_data = self.fetch_provider_models(provider)
                if not raw_data:
                    failed_providers.append(provider)
                    continue
                
                # Convert to structured format
                structured_models = self.convert_to_structured_format(provider, raw_data)
                if not structured_models:
                    failed_providers.append(provider)
                    continue
                
                # Create/update YAML file
                self.create_provider_yaml(provider, structured_models)
                updated_providers.append(provider)
                
            except Exception as e:
                logger.error(f"❌ Failed to update {provider}: {e}")
                failed_providers.append(provider)
        
        # Update master index
        if updated_providers:
            self.update_master_index(updated_providers)
        
        # Summary
        total_models = sum(len(yaml.safe_load(f.open())["models"]) 
                          for f in self.data_dir.glob("*_models_structured.yaml") 
                          if f.name != "00_master_index.yaml")
        
        print("\n" + "=" * 60)
        print("🎉 HELICONE DATABASE UPDATE COMPLETE")
        print("=" * 60)
        print(f"✅ Successfully updated: {len(updated_providers)} providers")
        print(f"❌ Failed to update: {len(failed_providers)} providers")
        print(f"📊 Total models in database: {total_models}")
        print(f"📁 Data directory: {self.data_dir}")
        print("=" * 60)
        
        if failed_providers:
            print(f"\n⚠️  Failed providers: {', '.join(failed_providers)}")
            print("💡 Try running again or check network connectivity")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Update model database from Helicone API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_from_helicone.py                           # Update all providers
  python update_from_helicone.py --providers openai        # Update OpenAI only
  python update_from_helicone.py --providers openai,anthropic,groq  # Multiple providers
  python update_from_helicone.py --dry-run                 # Show what would be updated
  python update_from_helicone.py --verbose                 # Detailed logging
        """
    )
    
    parser.add_argument(
        "--providers", 
        type=str,
        help="Comma-separated list of providers to update (default: all known providers)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse providers
    providers = None
    if args.providers:
        providers = [p.strip() for p in args.providers.split(",")]
    
    try:
        # Run updater
        updater = HeliconeUpdater()
        updater.update_providers(providers=providers, dry_run=args.dry_run)
        
    except KeyboardInterrupt:
        print("\n⏹️  Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()