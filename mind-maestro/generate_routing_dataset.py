#!/usr/bin/env python3
"""
Generate Dataset with Routing Agent Model Selections + Dynamic Parameters

This script processes all prompts from routerbench_english_only.csv through the 
routing agent to create a comprehensive dataset with:
- Original prompts
- Selected models 
- Model description vectors
- Dynamic routing parameters
"""

import csv
import json
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add the mind_maestro package to Python path
sys.path.append(str(Path(__file__).parent / "mind-maestro"))

from mind_maestro.routing_agent.router import ModelRouter
from mind_maestro.routing_agent.models import RoutingDecision, RoutingConfig
from mind_maestro.data.model_data import llm_benchmarks


def extract_dynamic_parameters(routing_decision: RoutingDecision) -> Dict[str, Any]:
    """Extract key dynamic parameters from routing decision for dataset."""
    
    # Core prompt analysis parameters
    prompt_analysis = routing_decision.prompt_analysis
    model_selection = routing_decision.model_selection
    
    dynamic_params = {
        # Task analysis
        "task_type": prompt_analysis.task_type.value,
        "complexity": prompt_analysis.complexity.value,
        "domain": prompt_analysis.domain.value,
        "context_length": prompt_analysis.context_length,
        "reasoning_steps_required": prompt_analysis.reasoning_steps_required,
        "requires_multimodal": prompt_analysis.requires_multimodal,
        
        # Analysis confidence and details
        "analysis_confidence": prompt_analysis.confidence_score,
        "keywords": prompt_analysis.keywords[:5] if prompt_analysis.keywords else [],
        
        # Model selection parameters
        "selection_confidence": model_selection.confidence_score,
        "estimated_performance": model_selection.estimated_performance,
        "efficiency_score": model_selection.selected_model.efficiency_score,
        "task_relevance_score": model_selection.selected_model.task_relevance_score,
        "cost_efficiency": model_selection.selected_model.cost_efficiency,
        
        # Selection reasoning (truncated)
        "selection_reasoning": model_selection.selection_reasoning[:200] + "..." if len(model_selection.selection_reasoning) > 200 else model_selection.selection_reasoning,
        
        # Alternative models considered
        "alternatives": [alt.name for alt in model_selection.alternatives[:3]],
        "fallback_model": model_selection.fallback_model.name if model_selection.fallback_model else None,
        
        # Workflow metrics
        "processing_time_ms": routing_decision.processing_time_ms,
        "routing_version": routing_decision.routing_version,
        "complexity_tier": getattr(routing_decision, 'complexity_tier', 'unknown'),
    }
    
    # Add AI workflow metrics if available
    if hasattr(routing_decision, 'ai_workflow_metrics') and routing_decision.ai_workflow_metrics:
        workflow = routing_decision.ai_workflow_metrics
        dynamic_params.update({
            "total_processing_time_ms": workflow.total_processing_time_ms,
            "total_tokens_consumed": workflow.total_tokens_consumed,
            "estimated_cost_usd": workflow.estimated_cost_usd,
            "overall_ai_confidence": workflow.overall_confidence,
            "openai_model_usage": workflow.openai_model_usage,
        })
    
    return dynamic_params


def get_model_description_vector(model_name: str, models_data: List[Dict]) -> List[float]:
    """Get the description vector for a model."""
    for model in models_data:
        if model["name"] == model_name:
            return model["description_vector"]
    
    # Fallback if model not found
    print(f"Warning: Model {model_name} not found in model data")
    return [0.0, 0.0, 0.0, 0.0, 0.0]


def process_batch(prompts_batch: List[Dict], router: ModelRouter, batch_num: int, total_batches: int) -> List[Dict]:
    """Process a batch of prompts through the routing agent."""
    results = []
    batch_size = len(prompts_batch)
    
    print(f"Processing batch {batch_num}/{total_batches} ({batch_size} prompts)...")
    
    for i, prompt_data in enumerate(prompts_batch):
        try:
            prompt_text = prompt_data["prompt"]
            
            # Route the prompt
            start_time = time.time()
            routing_decision = router.route_prompt(prompt_text)
            processing_time = (time.time() - start_time) * 1000
            
            # Extract data for dataset
            selected_model = routing_decision.model_selection.selected_model.name
            model_vector = get_model_description_vector(selected_model, llm_benchmarks["models"])
            dynamic_params = extract_dynamic_parameters(routing_decision)
            
            # Create result entry
            result = {
                "prompt": prompt_text,
                "prompt_id": prompt_data.get("prompt_id", f"prompt_{batch_num}_{i}"),
                "source": prompt_data.get("source", "routerbench"),
                "complexity_estimate": prompt_data.get("complexity_estimate", "unknown"),
                "selected_model": selected_model,
                "model_description_vector": model_vector,
                "dynamic_parameters": dynamic_params,
                "actual_processing_time_ms": processing_time
            }
            
            results.append(result)
            
            # Progress within batch
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{batch_size} in batch {batch_num}")
                
        except Exception as e:
            print(f"Error processing prompt in batch {batch_num}, index {i}: {str(e)}")
            # Add error entry to maintain dataset integrity
            results.append({
                "prompt": prompt_data.get("prompt", "ERROR_PROMPT"),
                "prompt_id": prompt_data.get("prompt_id", f"error_{batch_num}_{i}"),
                "source": prompt_data.get("source", "routerbench"),
                "complexity_estimate": prompt_data.get("complexity_estimate", "unknown"),
                "selected_model": "ERROR",
                "model_description_vector": [0.0, 0.0, 0.0, 0.0, 0.0],
                "dynamic_parameters": {"error": str(e)},
                "actual_processing_time_ms": 0.0
            })
    
    print(f"Completed batch {batch_num}/{total_batches}")
    return results


def load_routerbench_prompts(csv_path: str) -> List[Dict]:
    """Load prompts from routerbench CSV file."""
    prompts = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompts.append({
                "prompt": row["prompt"],
                "prompt_id": row.get("prompt_id", ""),
                "source": row.get("source", "routerbench"),
                "complexity_estimate": row.get("complexity_estimate", "unknown"),
                "length": int(row.get("length", 0)),
                "word_count": int(row.get("word_count", 0)),
            })
    
    return prompts


def save_results_batch(results: List[Dict], output_path: str, batch_num: int, is_first_batch: bool = False):
    """Save batch results to CSV file."""
    mode = 'w' if is_first_batch else 'a'
    
    with open(output_path, mode, newline='', encoding='utf-8') as file:
        fieldnames = [
            "prompt", "prompt_id", "source", "complexity_estimate",
            "selected_model", "model_description_vector", "dynamic_parameters",
            "actual_processing_time_ms"
        ]
        
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if is_first_batch:
            writer.writeheader()
        
        for result in results:
            # Convert lists and dicts to JSON strings for CSV storage
            result_copy = result.copy()
            result_copy["model_description_vector"] = json.dumps(result["model_description_vector"])
            result_copy["dynamic_parameters"] = json.dumps(result["dynamic_parameters"])
            
            writer.writerow(result_copy)
    
    print(f"Saved batch {batch_num} results to {output_path}")


def main():
    """Main function to generate the routing dataset."""
    
    # Configuration
    INPUT_CSV = "mind-maestro/mind_maestro/data/routerbench_english_only.csv"
    OUTPUT_CSV = "routerbench_with_model_selections.csv"
    BATCH_SIZE = 500  # Reduced batch size due to additional processing
    
    # Verify input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found")
        return
    
    print("🚀 Starting Dataset Generation with Routing Agent")
    print("=" * 60)
    
    # Initialize routing agent
    print("Initializing routing agent...")
    config = RoutingConfig()
    router = ModelRouter(config=config, use_embeddings=True)
    
    # Load prompts
    print(f"Loading prompts from {INPUT_CSV}...")
    prompts = load_routerbench_prompts(INPUT_CSV)
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts")
    
    # Calculate batching
    total_batches = (total_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {total_batches} batches of {BATCH_SIZE} prompts each")
    
    # Process in batches
    start_time = time.time()
    
    for batch_num in range(1, total_batches + 1):
        batch_start_idx = (batch_num - 1) * BATCH_SIZE
        batch_end_idx = min(batch_num * BATCH_SIZE, total_prompts)
        batch_prompts = prompts[batch_start_idx:batch_end_idx]
        
        # Process batch
        batch_results = process_batch(batch_prompts, router, batch_num, total_batches)
        
        # Save batch results
        save_results_batch(batch_results, OUTPUT_CSV, batch_num, is_first_batch=(batch_num == 1))
        
        # Progress reporting
        elapsed_time = time.time() - start_time
        processed_prompts = batch_end_idx
        prompts_per_second = processed_prompts / elapsed_time if elapsed_time > 0 else 0
        remaining_prompts = total_prompts - processed_prompts
        eta_seconds = remaining_prompts / prompts_per_second if prompts_per_second > 0 else 0
        
        print(f"Progress: {processed_prompts}/{total_prompts} ({processed_prompts/total_prompts*100:.1f}%)")
        print(f"Speed: {prompts_per_second:.2f} prompts/second")
        print(f"ETA: {eta_seconds/60:.1f} minutes")
        print("-" * 40)
    
    # Final statistics
    total_time = time.time() - start_time
    
    print("🎉 Dataset Generation Complete!")
    print("=" * 60)
    print(f"Total prompts processed: {total_prompts}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average speed: {total_prompts/total_time:.2f} prompts/second")
    
    # Get routing statistics
    stats = router.get_routing_statistics()
    print("\n📊 Routing Agent Statistics:")
    print(f"Most selected models: {list(stats['most_selected_models'].keys())[:5]}")
    print(f"Task type distribution: {stats['task_type_distribution']}")
    print(f"Complexity distribution: {stats['complexity_distribution']}")


if __name__ == "__main__":
    main()