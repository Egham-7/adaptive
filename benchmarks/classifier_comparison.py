import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Check NumPy version first
import numpy as np
if np.__version__.startswith('2.'):
    print("Warning: NumPy 2.x detected. Downgrading to NumPy 1.x for compatibility...")
    print("Please run: pip install 'numpy<2'")
    sys.exit(1)

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add the parent directories to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "prompt_task_complexity_classifier_quantized"))
sys.path.insert(0, str(project_root / "adaptive_ai"))

def get_classifier_functions() -> Tuple[Optional[Any], Optional[Any]]:
    """Get classifier functions directly."""
    quantized_classifier = None
    non_quantized_classifier = None
    
    # Try importing quantized classifier
    try:
        from prompt_task_complexity_classifier_quantized.src.prompt_classifier.classifier import QuantizedPromptClassifier
        model_path = project_root / "prompt_task_complexity_classifier_quantized/quantized_model_output"
        if not (model_path / "model_quantized.onnx").exists():
            print(f"❌ model_quantized.onnx not found at {model_path}!")
            print("Please ensure the model file exists at the specified path.")
        else:
            quantized_classifier = QuantizedPromptClassifier(model_path=str(model_path))
            print("✓ Successfully imported and initialized QuantizedPromptClassifier")
    except ImportError as e:
        print(f"✗ Failed to import QuantizedPromptClassifier: {e}")
    except Exception as e:
        print(f"✗ Failed to initialize QuantizedPromptClassifier: {e}")
    
    # Try importing non-quantized classifier
    try:
        from adaptive_ai.services.prompt_classifier import PromptClassifier
        non_quantized_classifier = PromptClassifier()
        print("✓ Successfully imported and initialized PromptClassifier")
    except ImportError as e:
        print(f"✗ Failed to import PromptClassifier: {e}")
    except Exception as e:
        print(f"✗ Failed to initialize PromptClassifier: {e}")
    
    return quantized_classifier, non_quantized_classifier

def benchmark_classifier(classifier: Any, prompts: List[str], num_runs: int = 5) -> Dict[str, float]:
    """Benchmark classifier performance."""
    times = []
    
    # Warmup
    for _ in range(2):
        if hasattr(classifier, 'classify_single_prompt'):
            for prompt in prompts[:10]:
                classifier.classify_single_prompt(prompt)
        else:
            classifier.classify_prompts(prompts[:10])
    
    # Benchmark
    for _ in range(num_runs):
        start_time = time.time()
        if hasattr(classifier, 'classify_single_prompt'):
            for prompt in prompts:
                classifier.classify_single_prompt(prompt)
        else:
            classifier.classify_prompts(prompts)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times_arr = np.array(times)
    return {
        "mean_time": float(np.mean(times_arr)),
        "std_time": float(np.std(times_arr)),
        "min_time": float(np.min(times_arr)),
        "max_time": float(np.max(times_arr)),
        "throughput": len(prompts) / np.mean(times_arr),
        "avg_per_prompt": np.mean(times_arr) / len(prompts)
    }

def process_prompts(prompts: List[str], quantized_classifier: Any, non_quantized_classifier: Any) -> Dict[str, List[Union[str, float, None]]]:
    """Process prompts using both classifiers and collect results."""
    results = {
        'prompts': prompts,
        'quantized_task_types': [],
        'non_quantized_task_types': [],
        'quantized_complexity': [],
        'non_quantized_complexity': [],
        'quantized_confidence': [],
        'non_quantized_confidence': []
    }
    
    # Initialize all result lists with None values
    for key in results:
        if key != 'prompts':
            results[key] = [None] * len(prompts)
    
    # Process with quantized classifier if available
    if quantized_classifier:
        try:
            print("Processing with quantized classifier...")
            # Process one prompt at a time to avoid shape issues
            for i, prompt in enumerate(prompts):
                try:
                    result = quantized_classifier.classify_single_prompt(prompt)
                    results['quantized_task_types'][i] = result['task_type_1']
                    results['quantized_complexity'][i] = result['prompt_complexity_score']
                    results['quantized_confidence'][i] = result['task_type_prob']
                except Exception as e:
                    print(f"✗ Error processing prompt '{prompt}': {e}")
            print("✓ Quantized classifier processing complete")
        except Exception as e:
            print(f"✗ Error processing with quantized classifier: {e}")
    
    # Process with non-quantized classifier if available
    if non_quantized_classifier:
        try:
            print("Processing with non-quantized classifier...")
            non_quantized_results = non_quantized_classifier.classify_prompts(prompts)
            for i, r in enumerate(non_quantized_results):
                results['non_quantized_task_types'][i] = r['task_type_1'][0] if isinstance(r['task_type_1'], list) else r['task_type_1']
                results['non_quantized_complexity'][i] = r['prompt_complexity_score'][0] if isinstance(r['prompt_complexity_score'], list) else r['prompt_complexity_score']
                results['non_quantized_confidence'][i] = r['task_type_prob'][0] if isinstance(r['task_type_prob'], list) else r['task_type_prob']
            print("✓ Non-quantized classifier processing complete")
        except Exception as e:
            print(f"✗ Error processing with non-quantized classifier: {e}")
    
    return results

def create_comparison_plots(results: Dict[str, List[Any]], output_dir: Path) -> None:
    """Create comparison plots for the results."""
    output_dir.mkdir(exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Filter out None values for plotting
    valid_data = df.dropna()
    
    if len(valid_data) == 0:
        print("Warning: No valid data for plotting")
        return
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    
    # 1. Task Type Agreement Plot
    if not valid_data['quantized_task_types'].isna().all() and not valid_data['non_quantized_task_types'].isna().all():
        plt.figure(figsize=(12, 6))
        agreement = (valid_data['quantized_task_types'] == valid_data['non_quantized_task_types']).mean() * 100
        plt.title(f'Task Type Agreement: {agreement:.2f}%')
        plt.pie([agreement, 100-agreement], labels=['Agree', 'Disagree'], autopct='%1.1f%%')
        plt.savefig(output_dir / 'task_type_agreement.png')
        plt.close()
        print("✓ Task type agreement plot saved")
    
    # 2. Complexity Score Comparison
    if not valid_data['quantized_complexity'].isna().all() and not valid_data['non_quantized_complexity'].isna().all():
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data['quantized_complexity'], valid_data['non_quantized_complexity'], alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect correlation line
        plt.xlabel('Quantized Complexity Score')
        plt.ylabel('Non-Quantized Complexity Score')
        plt.title('Complexity Score Comparison')
        plt.savefig(output_dir / 'complexity_comparison.png')
        plt.close()
        print("✓ Complexity comparison plot saved")
    
    # 3. Confidence Score Comparison
    if not valid_data['quantized_confidence'].isna().all() and not valid_data['non_quantized_confidence'].isna().all():
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data['quantized_confidence'], valid_data['non_quantized_confidence'], alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect correlation line
        plt.xlabel('Quantized Confidence')
        plt.ylabel('Non-Quantized Confidence')
        plt.title('Confidence Score Comparison')
        plt.savefig(output_dir / 'confidence_comparison.png')
        plt.close()
        print("✓ Confidence comparison plot saved")
    
    # 4. Task Type Distribution
    plt.figure(figsize=(15, 6))
    all_task_types = []
    if not valid_data['quantized_task_types'].isna().all():
        all_task_types.extend(valid_data['quantized_task_types'].dropna().tolist())
    if not valid_data['non_quantized_task_types'].isna().all():
        all_task_types.extend(valid_data['non_quantized_task_types'].dropna().tolist())
    
    if all_task_types:
        task_type_series = pd.Series(all_task_types)
        task_type_series.value_counts().plot(kind='bar')
        plt.title('Task Type Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'task_type_distribution.png')
        plt.close()
        print("✓ Task type distribution plot saved")

def create_similarity_analysis(results: Dict[str, List[Any]], output_dir: Path) -> None:
    """Create visualizations showing similarity between quantized and non-quantized results."""
    output_dir.mkdir(exist_ok=True)
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out None values
    valid_data = df.dropna()
    
    if len(valid_data) == 0:
        print("Warning: No valid data for similarity analysis")
        return
    
    # 1. Task Type Agreement Matrix
    plt.figure(figsize=(12, 8))
    task_types = pd.crosstab(
        valid_data['quantized_task_types'],
        valid_data['non_quantized_task_types'],
        normalize='index'
    )
    plt.imshow(task_types, cmap='YlOrRd')
    plt.colorbar(label='Agreement Rate')
    plt.title('Task Type Agreement Matrix')
    plt.xlabel('Non-Quantized Task Types')
    plt.ylabel('Quantized Task Types')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'task_type_agreement_matrix.png')
    plt.close()
    print("✓ Task type agreement matrix saved")
    
    # 2. Complexity Score Correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(
        valid_data['quantized_complexity'],
        valid_data['non_quantized_complexity'],
        alpha=0.5
    )
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Correlation')
    plt.xlabel('Quantized Complexity Score')
    plt.ylabel('Non-Quantized Complexity Score')
    plt.title('Complexity Score Correlation')
    correlation = valid_data['quantized_complexity'].corr(valid_data['non_quantized_complexity'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_correlation.png')
    plt.close()
    print("✓ Complexity correlation plot saved")
    
    # 3. Confidence Score Distribution
    plt.figure(figsize=(12, 6))
    plt.hist2d(
        valid_data['quantized_confidence'],
        valid_data['non_quantized_confidence'],
        bins=50,
        cmap='YlOrRd'
    )
    plt.colorbar(label='Count')
    plt.xlabel('Quantized Confidence')
    plt.ylabel('Non-Quantized Confidence')
    plt.title('Confidence Score Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png')
    plt.close()
    print("✓ Confidence distribution plot saved")
    
    # 4. Overall Agreement Statistics
    agreement_stats = {
        'Task Type Agreement': (valid_data['quantized_task_types'] == valid_data['non_quantized_task_types']).mean() * 100,
        'Complexity Correlation': correlation,
        'Confidence Correlation': valid_data['quantized_confidence'].corr(valid_data['non_quantized_confidence']),
        'Mean Complexity Difference': abs(valid_data['quantized_complexity'] - valid_data['non_quantized_complexity']).mean(),
        'Mean Confidence Difference': abs(valid_data['quantized_confidence'] - valid_data['non_quantized_confidence']).mean()
    }
    
    # Save agreement statistics
    with open(output_dir / 'agreement_statistics.json', 'w') as f:
        json.dump({k: float(v) for k, v in agreement_stats.items()}, f, indent=2)
    print("✓ Agreement statistics saved")

def main() -> None:
    print("=== Prompt Classifier Comparison Tool ===\n")
    
    # Get classifier functions
    try:
        quantized_classifier, non_quantized_classifier = get_classifier_functions()
    except Exception as e:
        print(f"Error getting classifier functions: {e}")
        return
    
    # Read prompts from CSV file
    csv_path = Path("./LmSys_analysis_results/lmsys_extracted_data.csv")
    max_prompts = 1000  # Strictly process only 10,000 prompts
    
    try:
        # Read exactly 10,000 prompts
        df = pd.read_csv(csv_path, nrows=max_prompts)
        test_prompts = df['input_prompt'].tolist()
        print(f"✓ Successfully loaded {len(test_prompts)} prompts from {csv_path}")
    except Exception as e:
        print(f"Error reading prompts from CSV: {e}")
        return
    
    # Process prompts
    print("\nProcessing test prompts...")
    try:
        results = process_prompts(test_prompts, quantized_classifier, non_quantized_classifier)
        print("✓ Prompt processing complete")
    except Exception as e:
        print(f"Error processing prompts: {e}")
        return
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    if quantized_classifier:
        print("\nQuantized Classifier Performance:")
        quantized_metrics = benchmark_classifier(quantized_classifier, test_prompts)
        print(json.dumps(quantized_metrics, indent=2))
    
    if non_quantized_classifier:
        print("\nNon-Quantized Classifier Performance:")
        non_quantized_metrics = benchmark_classifier(non_quantized_classifier, test_prompts)
        print(json.dumps(non_quantized_metrics, indent=2))
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    try:
        create_comparison_plots(results, Path("./comparison_plots"))
        print("✓ Plots created successfully")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Save results
    print("\nSaving results...")
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv("classifier_comparison_results.csv", index=False)
        print("✓ Results saved to classifier_comparison_results.csv")
        
        # Create similarity analysis
        print("\nCreating similarity analysis...")
        create_similarity_analysis(results, Path("./comparison_plots"))
        print("✓ Similarity analysis plots saved to ./comparison_plots/")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total prompts processed: {len(results['prompts'])}")
        
        if not all(x is None for x in results['quantized_task_types']):
            print(f"- Quantized classifier results: Available")
        else:
            print(f"- Quantized classifier results: Not available")
            
        if not all(x is None for x in results['non_quantized_task_types']):
            print(f"- Non-quantized classifier results: Available")
        else:
            print(f"- Non-quantized classifier results: Not available")
            
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 