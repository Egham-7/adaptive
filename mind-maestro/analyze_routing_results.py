#!/usr/bin/env python3
"""
Comprehensive Analysis of Routerbench Dataset with Model Selections

This script analyzes the generated routing dataset to provide insights into:
- Model selection patterns
- Task type distribution
- Performance metrics
- Dynamic parameter analysis
"""

import csv
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Add to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mind_maestro.data.model_data import llm_benchmarks


def load_and_parse_dataset(csv_path: str):
    """Load the dataset and parse JSON fields."""
    print(f"📊 Loading dataset from {csv_path}...")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for i, row in enumerate(reader):
            try:
                # Parse JSON fields
                row['model_description_vector'] = json.loads(row['model_description_vector'])
                row['dynamic_parameters'] = json.loads(row['dynamic_parameters'])
                row['actual_processing_time_ms'] = float(row['actual_processing_time_ms'])
                data.append(row)
                
                if (i + 1) % 5000 == 0:
                    print(f"  Loaded {i + 1} records...")
                    
            except Exception as e:
                print(f"Error parsing row {i}: {e}")
                continue
    
    print(f"✅ Successfully loaded {len(data)} records")
    return data


def analyze_model_selection_patterns(data):
    """Analyze which models were selected most frequently."""
    print("\n🎯 MODEL SELECTION ANALYSIS")
    print("=" * 50)
    
    # Count model selections
    model_counts = Counter(row['selected_model'] for row in data)
    total_selections = len(data)
    
    print(f"Most Selected Models:")
    for i, (model, count) in enumerate(model_counts.most_common(10), 1):
        percentage = (count / total_selections) * 100
        print(f"{i:2d}. {model:<20} {count:>6,} ({percentage:5.1f}%)")
    
    return model_counts


def analyze_task_type_distribution(data):
    """Analyze task type classification results."""
    print("\n📋 TASK TYPE ANALYSIS")
    print("=" * 50)
    
    # Extract task types from dynamic parameters
    task_types = []
    for row in data:
        if 'task_type' in row['dynamic_parameters']:
            task_types.append(row['dynamic_parameters']['task_type'])
        else:
            task_types.append('unknown')
    
    task_counts = Counter(task_types)
    total_tasks = len(task_types)
    
    print(f"Task Type Distribution:")
    for i, (task, count) in enumerate(task_counts.most_common(), 1):
        percentage = (count / total_tasks) * 100
        print(f"{i}. {task:<15} {count:>6,} ({percentage:5.1f}%)")
    
    return task_counts


def analyze_model_task_correlation(data):
    """Analyze which models are selected for which tasks."""
    print("\n🔗 MODEL-TASK CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Create model-task matrix
    model_task_matrix = defaultdict(lambda: defaultdict(int))
    
    for row in data:
        model = row['selected_model']
        task = row['dynamic_parameters'].get('task_type', 'unknown')
        model_task_matrix[model][task] += 1
    
    # Print top model-task combinations
    print("Top Model-Task Combinations:")
    all_combinations = []
    for model in model_task_matrix:
        for task, count in model_task_matrix[model].items():
            all_combinations.append((count, model, task))
    
    all_combinations.sort(reverse=True)
    
    for i, (count, model, task) in enumerate(all_combinations[:15], 1):
        print(f"{i:2d}. {model:<20} → {task:<12} ({count:>4,} selections)")
    
    return model_task_matrix


def analyze_complexity_patterns(data):
    """Analyze complexity classification and its effect on model selection."""
    print("\n⚡ COMPLEXITY ANALYSIS")
    print("=" * 50)
    
    # Extract complexity levels
    complexity_counts = Counter()
    complexity_models = defaultdict(Counter)
    
    for row in data:
        complexity = row['dynamic_parameters'].get('complexity', 'unknown')
        model = row['selected_model']
        
        complexity_counts[complexity] += 1
        complexity_models[complexity][model] += 1
    
    print("Complexity Distribution:")
    total = sum(complexity_counts.values())
    for complexity, count in complexity_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {complexity:<10} {count:>6,} ({percentage:5.1f}%)")
    
    print("\nTop Model Selections by Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        if complexity in complexity_models:
            print(f"\n  {complexity.upper()} Tasks:")
            top_models = complexity_models[complexity].most_common(5)
            for model, count in top_models:
                print(f"    {model:<20} {count:>4,}")
    
    return complexity_counts, complexity_models


def analyze_performance_metrics(data):
    """Analyze performance-related metrics from dynamic parameters."""
    print("\n📈 PERFORMANCE METRICS ANALYSIS")
    print("=" * 50)
    
    # Extract numeric metrics
    confidence_scores = []
    estimated_performance = []
    efficiency_scores = []
    processing_times = []
    
    for row in data:
        params = row['dynamic_parameters']
        
        if 'selection_confidence' in params:
            confidence_scores.append(params['selection_confidence'])
        if 'estimated_performance' in params:
            estimated_performance.append(params['estimated_performance'])
        if 'efficiency_score' in params:
            efficiency_scores.append(params['efficiency_score'])
        
        processing_times.append(row['actual_processing_time_ms'])
    
    def print_stats(name, values):
        if values:
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Median: {np.median(values):.3f}")
            print(f"  Std: {np.std(values):.3f}")
            print(f"  Min: {np.min(values):.3f}")
            print(f"  Max: {np.max(values):.3f}")
    
    print_stats("Selection Confidence", confidence_scores)
    print_stats("Estimated Performance", estimated_performance)
    print_stats("Efficiency Scores", efficiency_scores)
    print_stats("Processing Time (ms)", processing_times)
    
    return {
        'confidence': confidence_scores,
        'performance': estimated_performance,
        'efficiency': efficiency_scores,
        'processing_time': processing_times
    }


def analyze_model_vectors(data):
    """Analyze the model description vectors and their usage patterns."""
    print("\n🧮 MODEL VECTOR ANALYSIS")
    print("=" * 50)
    
    # Group by selected models and their vectors
    model_vectors = {}
    for row in data:
        model = row['selected_model']
        vector = row['model_description_vector']
        if model not in model_vectors:
            model_vectors[model] = vector
    
    print("Model Description Vectors [MMLU, HumanEval, MATH, GSM8K, Params(B)]:")
    
    # Sort by MMLU score (first element)
    sorted_models = sorted(model_vectors.items(), key=lambda x: x[1][0], reverse=True)
    
    for model, vector in sorted_models:
        mmlu, humaneval, math, gsm8k, params = vector
        print(f"  {model:<20} [{mmlu:5.1f}, {humaneval:5.1f}, {math:5.1f}, {gsm8k:5.1f}, {params:6.1f}B]")
    
    # Calculate model usage vs performance correlation
    model_counts = Counter(row['selected_model'] for row in data)
    
    print("\nModel Usage vs Performance (Top 5 by usage):")
    print("Model                Usage    MMLU  HumanEval  MATH  GSM8K  Params")
    for model, count in model_counts.most_common(5):
        if model in model_vectors:
            vector = model_vectors[model]
            usage_pct = (count / len(data)) * 100
            print(f"{model:<20} {usage_pct:5.1f}%   {vector[0]:5.1f}    {vector[1]:5.1f}  {vector[2]:5.1f}  {vector[3]:5.1f}  {vector[4]:6.1f}")
    
    return model_vectors


def generate_summary_insights(data, model_counts, task_counts):
    """Generate key insights and recommendations."""
    print("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)
    
    total_records = len(data)
    
    # Dataset overview
    print(f"📊 Dataset Overview:")
    print(f"  • Total prompts processed: {total_records:,}")
    print(f"  • Unique models used: {len(model_counts)}")
    print(f"  • Task types identified: {len(task_counts)}")
    
    # Model selection insights
    top_model = model_counts.most_common(1)[0]
    print(f"\n🎯 Model Selection Insights:")
    print(f"  • Most popular model: {top_model[0]} ({top_model[1]/total_records*100:.1f}%)")
    
    # Check for model diversity
    top_3_share = sum(count for _, count in model_counts.most_common(3)) / total_records
    print(f"  • Top 3 models handle: {top_3_share*100:.1f}% of all prompts")
    
    if top_3_share > 0.75:
        print("  ⚠️  High concentration in top models - consider routing diversity")
    else:
        print("  ✅ Good model distribution across different types")
    
    # Task distribution insights
    top_task = task_counts.most_common(1)[0]
    print(f"\n📋 Task Distribution Insights:")
    print(f"  • Most common task: {top_task[0]} ({top_task[1]/total_records*100:.1f}%)")
    
    # Performance insights
    print(f"\n⚡ Performance Insights:")
    avg_processing = np.mean([row['actual_processing_time_ms'] for row in data])
    print(f"  • Average processing time: {avg_processing:.1f}ms")
    print(f"  • Processing speed: ~{1000/avg_processing:.0f} prompts/second")
    
    # Recommendations
    print(f"\n🔄 Recommendations for Improvement:")
    print(f"  1. Validate model selections with human evaluation")
    print(f"  2. Implement actual AI routing agent (currently rule-based)")
    print(f"  3. Add cost-performance optimization metrics")
    print(f"  4. Consider model fine-tuning based on selection patterns")
    print(f"  5. Implement feedback loop for routing quality improvement")


def main():
    """Main analysis function."""
    # Configuration
    csv_path = "mind_maestro/data/routerbench_with_model_selections.csv"
    
    print("🚀 ROUTERBENCH DATASET ANALYSIS")
    print("=" * 60)
    print(f"Analyzing: {csv_path}")
    
    # Load and parse dataset
    data = load_and_parse_dataset(csv_path)
    
    if not data:
        print("❌ No data loaded. Exiting.")
        return
    
    # Run analyses
    model_counts = analyze_model_selection_patterns(data)
    task_counts = analyze_task_type_distribution(data)
    model_task_matrix = analyze_model_task_correlation(data)
    complexity_counts, complexity_models = analyze_complexity_patterns(data)
    performance_metrics = analyze_performance_metrics(data)
    model_vectors = analyze_model_vectors(data)
    
    # Generate insights
    generate_summary_insights(data, model_counts, task_counts)
    
    print(f"\n✨ Analysis Complete!")
    print(f"Dataset contains rich information about model routing patterns")
    print(f"Ready for machine learning model training and optimization")


if __name__ == "__main__":
    main()