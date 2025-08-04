#!/usr/bin/env python3
"""
Automated Model Configuration Pipeline

This script orchestrates the complete process of:
1. Extracting models from provider APIs
2. Analyzing capabilities with AI agent
3. Generating enriched YAML configurations
4. Creating Python provider configuration
5. Replacing manual model configurations

Usage:
    python automated_model_pipeline.py --full-pipeline
    python automated_model_pipeline.py --providers openai,anthropic --analyze-only
"""

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import click

# Add script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Load environment variables from .env file
env_file = script_dir / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

from extract_provider_models import ModelExtractor
from model_capability_agent import ModelCapabilityAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """Orchestrates the complete model configuration pipeline."""
    
    def __init__(
        self,
        base_dir: Path = Path("pipeline_output"),
        analysis_model: str = "gpt-4o-mini"
    ):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        
        # Pipeline directories
        self.raw_models_dir = self.base_dir / "raw_models"
        self.enriched_models_dir = self.base_dir / "enriched_models" 
        self.config_dir = self.base_dir / "generated_configs"
        
        # Create directories
        for dir_path in [self.raw_models_dir, self.enriched_models_dir, self.config_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.analysis_model = analysis_model
    
    async def extract_models(self, provider_filter: Optional[List[str]] = None) -> bool:
        """Step 1: Extract models from provider APIs."""
        logger.info("🚀 Step 1: Extracting models from provider APIs...")
        
        try:
            async with ModelExtractor(self.raw_models_dir) as extractor:
                provider_set = set(provider_filter) if provider_filter else None
                provider_models = await extractor.extract_all_models(provider_set)
                
                if not any(models for models in provider_models.values()):
                    logger.warning("No models extracted from any provider")
                    return False
                
                extractor.save_to_yaml(provider_models)
                
                total_models = sum(len(models) for models in provider_models.values())
                logger.info(f"✅ Extracted {total_models} models from {len(provider_models)} providers")
                return True
                
        except Exception as e:
            logger.error(f"❌ Model extraction failed: {e}")
            return False
    
    async def analyze_capabilities(self) -> bool:
        """Step 2: Analyze model capabilities with AI agent."""
        logger.info("🧠 Step 2: Analyzing model capabilities...")
        
        if not any(self.raw_models_dir.iterdir()):
            logger.error("No raw model data found. Run extraction first.")
            return False
        
        try:
            async with ModelCapabilityAgent(self.analysis_model) as agent:
                await agent.enrich_yaml_files(self.raw_models_dir)
                
                # Move enriched files to proper directory
                for provider_dir in self.raw_models_dir.iterdir():
                    if not provider_dir.is_dir():
                        continue
                    
                    enriched_files = list(provider_dir.glob("*_enriched_models.yaml"))
                    if enriched_files:
                        # Copy enriched files to enriched directory
                        target_dir = self.enriched_models_dir / provider_dir.name
                        target_dir.mkdir(exist_ok=True)
                        
                        for enriched_file in enriched_files:
                            shutil.copy2(enriched_file, target_dir / enriched_file.name)
                
                logger.info("✅ Model capability analysis complete")
                return True
                
        except Exception as e:
            logger.error(f"❌ Capability analysis failed: {e}")
            return False
    
    def generate_configurations(self) -> bool:
        """Step 3: Generate Python configuration files."""
        logger.info("⚙️  Step 3: Generating configuration files...")
        
        if not any(self.enriched_models_dir.iterdir()):
            logger.error("No enriched model data found. Run analysis first.")
            return False
        
        try:
            # Generate provider configuration
            agent = ModelCapabilityAgent(self.analysis_model)
            config_file = self.config_dir / "providers.py"
            agent.generate_provider_config(self.enriched_models_dir, config_file)
            
            # Generate task mappings
            self._generate_task_mappings()
            
            # Generate summary report
            self._generate_summary_report()
            
            logger.info("✅ Configuration generation complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration generation failed: {e}")
            return False
    
    def _generate_task_mappings(self) -> None:
        """Generate task mapping configuration."""
        import yaml
        
        task_mappings = {}
        
        # Collect task capabilities from all providers
        for provider_dir in self.enriched_models_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            
            enriched_files = list(provider_dir.glob("*_enriched_models.yaml"))
            if not enriched_files:
                continue
            
            with open(enriched_files[0], 'r') as f:
                data = yaml.safe_load(f)
            
            models = data.get("models", [])
            for model in models:
                task_capabilities = model.get("task_capabilities", {})
                
                for task_type, capability in task_capabilities.items():
                    if task_type not in task_mappings:
                        task_mappings[task_type] = {
                            "description": f"Models optimized for {task_type.replace('_', ' ')}",
                            "preferred_models": [],
                            "complexity_levels": set(),
                            "recommended_params": {}
                        }
                    
                    # Add model if it has good suitability score
                    suitability = capability.get("suitability_score", 0.0)
                    if suitability >= 0.7:
                        model_entry = {
                            "model_id": model["id"],
                            "provider": model["provider"],
                            "suitability_score": suitability,
                            "complexity_levels": capability.get("complexity_levels", [])
                        }
                        task_mappings[task_type]["preferred_models"].append(model_entry)
                    
                    # Update complexity levels
                    complexity_levels = capability.get("complexity_levels", [])
                    task_mappings[task_type]["complexity_levels"].update(complexity_levels)
                    
                    # Update recommended params
                    recommended_params = capability.get("recommended_params", {})
                    if recommended_params:
                        task_mappings[task_type]["recommended_params"] = recommended_params
        
        # Sort models by suitability score
        for task_data in task_mappings.values():
            task_data["preferred_models"].sort(
                key=lambda x: x["suitability_score"], 
                reverse=True
            )
            task_data["complexity_levels"] = sorted(list(task_data["complexity_levels"]))
        
        # Save task mappings
        task_mappings_file = self.config_dir / "task_mappings.yaml"
        with open(task_mappings_file, 'w') as f:
            yaml.dump(task_mappings, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated task mappings: {task_mappings_file}")
    
    def _generate_summary_report(self) -> None:
        """Generate a summary report of the pipeline results."""
        import yaml
        from datetime import datetime
        
        summary = {
            "pipeline_summary": {
                "generated_at": datetime.now().isoformat(),
                "analysis_model": self.analysis_model,
                "total_providers": 0,
                "total_models": 0,
                "task_types_analyzed": set(),
                "complexity_levels": set()
            },
            "providers": {},
            "task_coverage": {},
            "recommendations": []
        }
        
        # Analyze enriched models
        for provider_dir in self.enriched_models_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            
            enriched_files = list(provider_dir.glob("*_enriched_models.yaml"))
            if not enriched_files:
                continue
            
            with open(enriched_files[0], 'r') as f:
                data = yaml.safe_load(f)
            
            provider_id = data["provider"]["id"]
            models = data.get("models", [])
            
            summary["pipeline_summary"]["total_providers"] += 1
            summary["pipeline_summary"]["total_models"] += len(models)
            
            # Provider summary
            provider_summary = {
                "name": data["provider"]["name"],
                "total_models": len(models),
                "average_quality_score": 0.0,
                "specializations": set(),
                "model_tiers": {}
            }
            
            quality_scores = []
            for model in models:
                performance = model.get("performance_profile", {})
                quality_score = performance.get("estimated_quality_score", 0.0)
                if quality_score > 0:
                    quality_scores.append(quality_score)
                
                # Track specializations
                specializations = performance.get("specializations", [])
                provider_summary["specializations"].update(specializations)
                
                # Track model tiers
                complexity = performance.get("overall_complexity", "medium")
                if complexity not in provider_summary["model_tiers"]:
                    provider_summary["model_tiers"][complexity] = 0
                provider_summary["model_tiers"][complexity] += 1
                
                # Track task types
                task_capabilities = model.get("task_capabilities", {})
                summary["pipeline_summary"]["task_types_analyzed"].update(task_capabilities.keys())
                
                for capability in task_capabilities.values():
                    complexity_levels = capability.get("complexity_levels", [])
                    summary["pipeline_summary"]["complexity_levels"].update(complexity_levels)
            
            if quality_scores:
                provider_summary["average_quality_score"] = sum(quality_scores) / len(quality_scores)
            
            provider_summary["specializations"] = list(provider_summary["specializations"])
            summary["providers"][provider_id] = provider_summary
        
        # Convert sets to lists for YAML serialization
        summary["pipeline_summary"]["task_types_analyzed"] = sorted(list(summary["pipeline_summary"]["task_types_analyzed"]))
        summary["pipeline_summary"]["complexity_levels"] = sorted(list(summary["pipeline_summary"]["complexity_levels"]))
        
        # Generate recommendations
        recommendations = []
        
        if summary["pipeline_summary"]["total_models"] > 50:
            recommendations.append("Consider implementing model filtering to focus on most relevant models")
        
        if len(summary["providers"]) > 5:
            recommendations.append("Implement provider fallback strategies for better reliability")
        
        recommendations.append("Regularly re-run analysis to capture new models and capability changes")
        recommendations.append("Consider integrating benchmark scores for more objective capability assessment")
        
        summary["recommendations"] = recommendations
        
        # Save summary
        summary_file = self.config_dir / "pipeline_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated pipeline summary: {summary_file}")
    
    async def run_full_pipeline(self, provider_filter: Optional[List[str]] = None) -> bool:
        """Run the complete pipeline."""
        logger.info("🚀 Starting full model configuration pipeline...")
        
        steps = [
            ("Extract Models", self.extract_models, provider_filter),
            ("Analyze Capabilities", self.analyze_capabilities, None),
            ("Generate Configurations", self.generate_configurations, None)
        ]
        
        for step_name, step_func, args in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {step_name}")
            logger.info('='*60)
            
            if asyncio.iscoroutinefunction(step_func):
                if args is not None:
                    success = await step_func(args)
                else:
                    success = await step_func()
            else:
                if args is not None:
                    success = step_func(args)
                else:
                    success = step_func()
            
            if not success:
                logger.error(f"❌ Pipeline failed at step: {step_name}")
                return False
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Pipeline completed successfully!")
        logger.info('='*60)
        
        self._print_results_summary()
        return True
    
    def _print_results_summary(self) -> None:
        """Print summary of pipeline results."""
        print(f"\n📁 Pipeline Output Structure:")
        print(f"  {self.base_dir}/")
        print(f"  ├── raw_models/           # Extracted model data")
        print(f"  ├── enriched_models/      # AI-analyzed capabilities")
        print(f"  └── generated_configs/    # Python configurations")
        print(f"      ├── providers.py      # Provider configuration")
        print(f"      ├── task_mappings.yaml # Task-specific mappings")
        print(f"      └── pipeline_summary.yaml # Analysis summary")
        
        print(f"\n🔄 Next Steps:")
        print(f"  1. Review generated configurations in {self.config_dir}")
        print(f"  2. Replace adaptive_ai/config/providers.py with generated version")
        print(f"  3. Update task mappings in adaptive_ai/config/task_mappings.py")
        print(f"  4. Test the new configuration with your AI service")
        
        print(f"\n⚠️  Important:")
        print(f"  - Review AI-generated capability assessments for accuracy")
        print(f"  - Validate task type mappings against your use cases")
        print(f"  - Consider running benchmarks for objective validation")


@click.command()
@click.option(
    "--providers",
    default=None,
    help="Comma-separated list of providers to process (e.g., 'openai,anthropic')"
)
@click.option(
    "--output-dir",
    default="pipeline_output",
    help="Output directory for pipeline results"
)
@click.option(
    "--analysis-model",
    default="gpt-4o-mini",
    help="Model to use for capability analysis"
)
@click.option(
    "--extract-only",
    is_flag=True,
    help="Only extract models, skip analysis"
)
@click.option(
    "--analyze-only",
    is_flag=True,
    help="Only analyze existing extracted models"
)
@click.option(
    "--full-pipeline",
    is_flag=True,
    help="Run complete pipeline (default if no other options specified)"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    providers: Optional[str],
    output_dir: str,
    analysis_model: str,
    extract_only: bool,
    analyze_only: bool,
    full_pipeline: bool,
    verbose: bool
) -> None:
    """Automated pipeline for model extraction, analysis, and configuration generation."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse providers
    provider_filter = None
    if providers:
        provider_filter = [p.strip() for p in providers.split(",")]
        logger.info(f"Processing providers: {provider_filter}")
    
    # Default to full pipeline if no specific option
    if not any([extract_only, analyze_only]):
        full_pipeline = True
    
    pipeline = ModelPipeline(Path(output_dir), analysis_model)
    
    async def run_pipeline():
        try:
            if full_pipeline:
                success = await pipeline.run_full_pipeline(provider_filter)
            elif extract_only:
                logger.info("Running extraction only...")
                success = await pipeline.extract_models(provider_filter)
            elif analyze_only:
                logger.info("Running analysis only...")
                success = await pipeline.analyze_capabilities()
                if success:
                    success = pipeline.generate_configurations()
            else:
                logger.error("No operation specified")
                return False
            
            return success
            
        except KeyboardInterrupt:
            logger.info("Pipeline cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(run_pipeline())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()