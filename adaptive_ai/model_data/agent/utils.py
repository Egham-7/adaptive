"""
Agent Utilities

Helper functions for YAML handling, caching, and progress tracking.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any

import config
import yaml

logger = logging.getLogger(__name__)


def load_models_to_process() -> list[tuple[str, dict[str, Any], str, str]]:
    """Load models that need enrichment - simple field-based check, no caching"""
    models_to_process = []

    # Only process these providers as requested
    target_providers = {"ANTHROPIC", "GROK", "GROQ", "GOOGLE", "DEEPSEEK", "OPENAI"}

    # Load models from structured YAML files (only target providers)
    structured_path = Path(config.STRUCTURED_MODELS_PATH)
    target_files: list[Any] = []
    for provider in target_providers:
        target_files.extend(
            structured_path.glob(f"{provider.lower()}_models_structured.yaml")
        )

    print(f"🎯 Processing only these providers: {', '.join(target_providers)}")

    for yaml_file in target_files:
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            provider = data["provider_info"]["name"]
            print(f"📂 Loading {provider} models from {yaml_file.name}...")
            
            empty_count = 0
            filled_count = 0

            for model_key, model_data in data.get("models", {}).items():
                # Check if this model has ANY empty fields that need enrichment
                empty_fields = []
                
                if not model_data.get("description") or model_data.get("description") == "":
                    empty_fields.append("description")
                if model_data.get("max_context_tokens") is None:
                    empty_fields.append("max_context_tokens")
                if model_data.get("max_output_tokens") is None:
                    empty_fields.append("max_output_tokens")
                if not model_data.get("task_type") or model_data.get("task_type") == "":
                    empty_fields.append("task_type")
                if not model_data.get("complexity") or model_data.get("complexity") == "":
                    empty_fields.append("complexity")
                if model_data.get("supports_function_calling") is None:
                    empty_fields.append("supports_function_calling")
                if not model_data.get("model_size_params") or model_data.get("model_size_params") == "":
                    empty_fields.append("model_size_params")
                if not model_data.get("latency_tier") or model_data.get("latency_tier") == "":
                    empty_fields.append("latency_tier")
                
                # Only process if there are empty fields
                if empty_fields:
                    models_to_process.append(
                        (provider, model_data, str(yaml_file), model_key)
                    )
                    empty_count += 1
                    print(f"  📝 {model_data['model_name']}: needs {', '.join(empty_fields)}")
                else:
                    filled_count += 1
            
            print(f"  ✅ {provider}: {filled_count} already enriched, {empty_count} need enrichment")

        except Exception as e:
            logger.error(f"Error loading {yaml_file}: {e}")

    logger.info(
        f"Found {len(models_to_process)} models needing enrichment from target providers"
    )
    return models_to_process


def update_yaml_file(
    yaml_file: str, model_key: str, extracted_info: dict[str, Any]
) -> bool:
    """Update YAML file with extracted information (fill ALL empty fields)"""
    try:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if "models" not in data or model_key not in data["models"]:
            logger.error(f"Model key {model_key} not found in {yaml_file}")
            return False

        model_entry = data["models"][model_key]

        # Update ALL empty fields (preserve existing data)
        if not model_entry.get("description") and extracted_info.get("description"):
            model_entry["description"] = extracted_info["description"]

        if model_entry.get("max_context_tokens") is None and extracted_info.get(
            "max_context_tokens"
        ):
            model_entry["max_context_tokens"] = extracted_info["max_context_tokens"]

        if model_entry.get("max_output_tokens") is None and extracted_info.get(
            "max_output_tokens"
        ):
            model_entry["max_output_tokens"] = extracted_info["max_output_tokens"]

        if (
            model_entry.get("supports_function_calling") is None
            and extracted_info.get("supports_function_calling") is not None
        ):
            model_entry["supports_function_calling"] = extracted_info[
                "supports_function_calling"
            ]

        if not model_entry.get("task_type") and extracted_info.get("task_type"):
            model_entry["task_type"] = extracted_info["task_type"]

        if not model_entry.get("complexity") and extracted_info.get("complexity"):
            model_entry["complexity"] = extracted_info["complexity"]

        # Fill additional empty fields
        if not model_entry.get("model_size_params") and extracted_info.get(
            "model_size_params"
        ):
            model_entry["model_size_params"] = extracted_info["model_size_params"]

        if not model_entry.get("latency_tier") and extracted_info.get("latency_tier"):
            model_entry["latency_tier"] = extracted_info["latency_tier"]

        # Ensure languages_supported is a list if provided
        if not model_entry.get("languages_supported") and extracted_info.get(
            "languages_supported"
        ):
            model_entry["languages_supported"] = extracted_info["languages_supported"]
        elif not model_entry.get("languages_supported"):
            model_entry["languages_supported"] = []  # Default to empty list

        # Add enrichment metadata
        model_entry["_enrichment"] = {
            "enriched": True,
            "method": "langgraph_workflow",
            "confidence_score": extracted_info.get("confidence_score", 0.0),
            "enriched_at": time.time(),
        }

        # Save updated YAML
        with open(yaml_file, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

        return True

    except Exception as e:
        logger.error(f"Error updating {yaml_file}: {e}")
        return False


def save_processed_cache(processed_models: set[str], failed_models: list[str]) -> None:
    """Save processing cache"""
    cache_file = Path(config.CACHE_PATH) / "langgraph_processed_models.json"
    cache_file.parent.mkdir(exist_ok=True)

    try:
        data = {
            "processed": list(processed_models),
            "failed": failed_models,
            "last_updated": time.time(),
            "method": "langgraph_workflow",
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save cache: {e}")


class ProcessingTracker:
    """Track processing progress and results"""

    def __init__(self) -> None:
        self.processed_models: set[str] = set()
        self.failed_models: list[str] = []
        self.successful_count = 0
        self.failed_count = 0
        self.start_time = time.time()

    def mark_success(self, provider: str, model_name: str) -> None:
        """Mark a model as successfully processed"""
        model_id = f"{provider}:{model_name}"
        self.processed_models.add(model_id)
        self.successful_count += 1
        logger.info(f"✅ Successfully enriched {model_id}")

    def mark_failure(self, provider: str, model_name: str) -> None:
        """Mark a model as failed"""
        model_id = f"{provider}:{model_name}"
        self.failed_models.append(model_id)
        self.failed_count += 1
        logger.warning(f"❌ Failed to enrich {model_id}")

    def print_progress(
        self, current: int, total: int
    ) -> None:
        """Print progress update"""
        progress_pct = (current / total) * 100 if total > 0 else 0

        print("\\n📈 PROGRESS UPDATE")
        print(f"   Progress: {current}/{total} ({progress_pct:.1f}%)")
        print(f"   Successful: {self.successful_count}")
        print(f"   Failed: {self.failed_count}")

    def print_final_summary(self) -> None:
        """Print final processing summary"""
        elapsed_time = time.time() - self.start_time
        total_processed = self.successful_count + self.failed_count
        success_rate = (
            (self.successful_count / total_processed * 100)
            if total_processed > 0
            else 0
        )

        print("\\n" + "=" * 60)
        print("🎉 LANGGRAPH ENRICHMENT COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully enriched: {self.successful_count} models")
        print(f"❌ Failed to enrich: {self.failed_count} models")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"⏱️  Processing time: {elapsed_time/60:.1f} minutes")
        print("=" * 60)
