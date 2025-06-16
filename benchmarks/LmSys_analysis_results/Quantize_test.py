import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# First, let's handle the NumPy compatibility issue
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("Warning: NumPy 2.x detected. You may need to downgrade to NumPy 1.x for compatibility.")
        print("Run: pip install 'numpy<2'")
except ImportError:
    print("NumPy not found. Installing...")

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Required dependencies:
# 1. Install main dependencies:
#    pip install 'numpy<2' torch==2.2.0 matplotlib pandas tqdm transformers
#
# 2. Fix the adaptive_ai import path (see instructions below)

# Add the parent directories to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

# Alternative import strategies - try multiple paths
def import_classifiers():
    """Try different import strategies for the classifiers."""
    quantized_classifier = None
    non_quantized_classifier = None
    
    # Strategy 1: Try the original import
    try:
        from prompt_task_complexity_classifier_quantized.src.prompt_classifier.classifier import QuantizedPromptClassifier
        quantized_classifier = QuantizedPromptClassifier
        print("✓ Successfully imported QuantizedPromptClassifier")
    except ImportError as e:
        print(f"✗ Failed to import QuantizedPromptClassifier: {e}")
    
    # Strategy 2: Try different adaptive_ai import paths
    adaptive_ai_import_attempts = [
        "adaptive_ai.services.prompt_classifier",
        "adaptive_ai.adaptive_ai.services.prompt_classifier", 
        "services.prompt_classifier",
        "src.services.prompt_classifier"
    ]
    
    for import_path in adaptive_ai_import_attempts:
        try:
            module = __import__(import_path, fromlist=['PromptClassifier'])
            non_quantized_classifier = module.PromptClassifier
            print(f"✓ Successfully imported PromptClassifier from {import_path}")
            break
        except ImportError as e:
            print(f"✗ Failed to import from {import_path}: {e}")
            continue
    
    return quantized_classifier, non_quantized_classifier

INPUT_CSV = Path("./lmsys_extracted_data.csv")
OUTPUT_CSV = Path("./adaptive_benchmark_results.csv")
MAX_ROWS = 1000  # Only process first 1000 rows

# Model paths - update these if your models are located elsewhere
QUANTIZED_MODEL_PATH = "./models/quantized"
# Alternative paths to try if the default doesn't work
ALTERNATIVE_QUANTIZED_PATHS = [
    "./quantized_models",
    "./models/quantized_model",
    "../models/quantized",
    "./prompt_task_complexity_classifier_quantized/models/quantized"
]

def load_classifiers():
    """Initialize both quantized and non-quantized classifiers."""
    QuantizedPromptClassifier, PromptClassifier = import_classifiers()
    
    quantized_classifier = None
    non_quantized_classifier = None
    
    if QuantizedPromptClassifier:
        # Try multiple paths for the quantized model
        paths_to_try = [QUANTIZED_MODEL_PATH] + ALTERNATIVE_QUANTIZED_PATHS
        
        for path in paths_to_try:
            try:
                print(f"Trying to load quantized model from: {path}")
                quantized_classifier = QuantizedPromptClassifier.from_pretrained(path)
                print(f"✓ Quantized classifier loaded successfully from {path}")
                break
            except Exception as e:
                print(f"✗ Failed to load from {path}: {e}")
                continue
        
        if not quantized_classifier:
            print("✗ Could not load quantized classifier from any path")
            print("Available paths tried:")
            for path in paths_to_try:
                exists = Path(path).exists()
                print(f"  - {path}: {'exists' if exists else 'not found'}")
    
    if PromptClassifier:
        try:
            non_quantized_classifier = PromptClassifier()
            print("✓ Non-quantized classifier loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load non-quantized classifier: {e}")
    
    if not quantized_classifier and not non_quantized_classifier:
        raise RuntimeError("Failed to load any classifiers. Please check your setup.")
    
    return quantized_classifier, non_quantized_classifier

def process_prompts(prompts: List[str], quantized_classifier, non_quantized_classifier) -> Dict[str, List[Any]]:
    """Process prompts using available classifiers."""
    results = {
        'prompts': prompts,
        'quantized_task_types': [],
        'non_quantized_task_types': [],
        'quantized_complexity': [],
        'non_quantized_complexity': [],
        'quantized_confidence': [],
        'non_quantized_confidence': []
    }
    
    # Process with quantized classifier if available
    if quantized_classifier:
        try:
            print("Processing with quantized classifier...")
            quantized_results = quantized_classifier.classify_prompts(prompts)
            results['quantized_task_types'] = [r['task_type_1'] for r in quantized_results]
            results['quantized_complexity'] = [r['prompt_complexity_score'] for r in quantized_results]
            results['quantized_confidence'] = [r['task_type_prob'] for r in quantized_results]
            print("✓ Quantized classifier processing complete")
        except Exception as e:
            print(f"✗ Error processing with quantized classifier: {e}")
            # Fill with None values
            results['quantized_task_types'] = [None] * len(prompts)
            results['quantized_complexity'] = [None] * len(prompts)
            results['quantized_confidence'] = [None] * len(prompts)
    else:
        # Fill with None values if classifier not available
        results['quantized_task_types'] = [None] * len(prompts)
        results['quantized_complexity'] = [None] * len(prompts)
        results['quantized_confidence'] = [None] * len(prompts)
    
    # Process with non-quantized classifier if available
    if non_quantized_classifier:
        try:
            print("Processing with non-quantized classifier...")
            non_quantized_results = non_quantized_classifier.classify_prompts(prompts)
            results['non_quantized_task_types'] = [r['task_type_1'][0] if isinstance(r['task_type_1'], list) else r['task_type_1'] for r in non_quantized_results]
            results['non_quantized_complexity'] = [r['prompt_complexity_score'][0] if isinstance(r['prompt_complexity_score'], list) else r['prompt_complexity_score'] for r in non_quantized_results]
            results['non_quantized_confidence'] = [r['task_type_prob'][0] if isinstance(r['task_type_prob'], list) else r['task_type_prob'] for r in non_quantized_results]
            print("✓ Non-quantized classifier processing complete")
        except Exception as e:
            print(f"✗ Error processing with non-quantized classifier: {e}")
            # Fill with None values
            results['non_quantized_task_types'] = [None] * len(prompts)
            results['non_quantized_complexity'] = [None] * len(prompts)
            results['non_quantized_confidence'] = [None] * len(prompts)
    else:
        # Fill with None values if classifier not available
        results['non_quantized_task_types'] = [None] * len(prompts)
        results['non_quantized_complexity'] = [None] * len(prompts)
        results['non_quantized_confidence'] = [None] * len(prompts)
    
    return results

def create_comparison_plots(results: Dict[str, List[Any]], output_dir: Path):
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
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # 1. Task Type Agreement Plot (only if both classifiers have data)
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

def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking dependencies...")
    
    required_packages = ['torch', 'transformers', 'matplotlib', 'pandas', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    print("=== Adaptive AI Classifier Benchmark ===\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Check if input file exists
    if not INPUT_CSV.exists():
        print(f"Error: Input file {INPUT_CSV} not found.")
        print("Please ensure the CSV file exists in the current directory.")
        return
    
    # Read input CSV
    print(f"Reading input CSV from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        df = df.head(MAX_ROWS)  # Only take first 1000 rows
        print(f"✓ Loaded {len(df)} rows from CSV")
        
        if 'input_prompt' not in df.columns:
            print("Error: 'input_prompt' column not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Load classifiers
    print("\nLoading classifiers...")
    try:
        quantized_classifier, non_quantized_classifier = load_classifiers()
    except Exception as e:
        print(f"Error loading classifiers: {e}")
        print("\nTroubleshooting steps:")
        print("1. Downgrade NumPy: pip install 'numpy<2'")
        print("2. Check that the adaptive_ai package is properly installed")
        print("3. Verify the quantized model path exists: ./models/quantized")
        return
    
    # Process prompts
    print("\nProcessing prompts...")
    try:
        prompts = df['input_prompt'].tolist()
        results = process_prompts(prompts, quantized_classifier, non_quantized_classifier)
        print("✓ Prompt processing complete")
    except Exception as e:
        print(f"Error processing prompts: {e}")
        return
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    try:
        create_comparison_plots(results, Path("./comparison_plots"))
        print("✓ Plots created successfully")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Save detailed results
    print("\nSaving results...")
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ Results saved to {OUTPUT_CSV}")
        print("✓ Comparison plots saved to ./comparison_plots/")
        
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