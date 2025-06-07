from datasets import load_dataset, concatenate_datasets
from collections import Counter
import requests
import numpy as np
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Load and combine all splits
ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
all_examples = concatenate_datasets([ds["train"], ds["test"], ds["validation"]])

# Take only first 100 examples for testing
all_examples = all_examples.select(range(100))

print("\nProcessing first 100 examples from the dataset")

print(f"\nTotal examples across all splits: {len(all_examples)}")


def process_single_example(example, i):
    API_URL = "http://localhost:8000/predict"

    # Construct the prompt from the question and choices
    question = example["question"]
    choices = example["choices"]["text"]
    choices_text = "\n".join(
        [f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)]
    )
    prompt = f"{question}\n\n{choices_text}"

    request_data = {"prompt": prompt}
    try:
        response = requests.post(API_URL, json=request_data)
        result = response.json()

        if "weighted_vector" not in result:
            return {"failed": True, "index": i, "prompt": prompt, "result": result}

        return {
            "failed": False,
            "vector": result["weighted_vector"],
            "task_type": result.get("task_type"),
            "index": i,
        }
    except Exception as e:
        return {
            "failed": True,
            "index": i,
            "error": str(e),
            "response": getattr(e, "response", None),
        }


# Function to process examples
def process_examples(examples) -> Dict[str, Any]:
    vectors = []
    failed_questions: List[int] = []
    task_type_counts: Counter = Counter()

    print("\nAnalyzing API responses for questions...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_example = {
            executor.submit(process_single_example, example, i): i
            for i, example in enumerate(examples)
        }

        # Process results as they complete
        for future in tqdm(
            as_completed(future_to_example),
            total=len(examples),
            desc="Processing questions",
        ):
            result = future.result()

            if result["failed"]:
                if len(failed_questions) < 5:  # Only show details for first 5 failures
                    print(f"\nFailed question {result['index']}:")
                    if "prompt" in result:
                        print(f"Question: {result['prompt']}")
                        print(
                            f"Full API Response: {json.dumps(result['result'], indent=2)}"
                        )
                    if "error" in result:
                        print(f"Error: {result['error']}")
                        if result["response"]:
                            print(f"Response status: {result['response'].status_code}")
                            print(f"Response text: {result['response'].text}")
                failed_questions.append(result["index"])
            else:
                vectors.append(result["vector"])
                if result["task_type"]:
                    task_type_counts[result["task_type"]] += 1

    print(f"\nTotal successful vectors: {len(vectors)}")
    print(f"Total failed questions: {len(failed_questions)}")
    print(
        f"Failed question indices: {failed_questions[:10]}..."
    )  # Show first 10 failed indices

    print("\nTask type distribution:")
    for task_type, count in task_type_counts.items():
        print(f"{task_type}: {count}")

    return {
        "vectors": vectors,
        "failed_questions": failed_questions,
        "task_type_counts": task_type_counts,
    }


# Process all examples
results = process_examples(all_examples)


# Function to calculate statistics for vectors
def calculate_vector_stats(
    vectors: List[List[float]], components: List[str]
) -> Dict[str, Dict[str, float]]:
    if not vectors:
        return {}

    vectors_array = np.array(vectors)
    stats = {}

    for i, comp in enumerate(components):
        values = vectors_array[:, i]
        stats[comp] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q1": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "q3": float(np.percentile(values, 75)),
        }

    return stats


# Components for vector analysis
components = [
    "creativity_scope",
    "reasoning",
    "constraint_ct",
    "contextual_knowledge",
    "domain_knowledge",
]

# Calculate statistics
stats = calculate_vector_stats(results["vectors"], components)

# Create visualizations directory if it doesn't exist
visualizations_dir = Path("visualizations")
visualizations_dir.mkdir(exist_ok=True)

try:
    # 1. Distribution plots for each component
    plt.figure(figsize=(15, 10))
    vectors_array = np.array(results["vectors"])
    for i, comp in enumerate(components):
        plt.subplot(2, 3, i + 1)
        sns.histplot(vectors_array[:, i], kde=True)
        plt.title(f"{comp} Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "component_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Box plots for all components
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(vectors_array, columns=components))
    plt.title("Component Value Distributions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "component_boxplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = pd.DataFrame(vectors_array, columns=components).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Component Correlations")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Task type distribution pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        results["task_type_counts"].values(),
        labels=results["task_type_counts"].keys(),
        autopct="%1.1f%%",
        textprops={"fontsize": 10},
    )
    plt.title("Task Type Distribution")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "task_type_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Additional: Scatter plot matrix for components
    plt.figure(figsize=(15, 15))
    sns.pairplot(pd.DataFrame(vectors_array, columns=components))
    plt.savefig(
        visualizations_dir / "component_scatter_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        "\nAll visualizations have been successfully saved to the 'visualizations' directory"
    )
    print("Generated plots:")
    print("1. component_distributions.png - Distribution of each component")
    print("2. component_boxplots.png - Box plots showing statistical distributions")
    print("3. correlation_heatmap.png - Correlation between components")
    print("4. task_type_distribution.png - Distribution of task types")
    print("5. component_scatter_matrix.png - Scatter plot matrix of components")

except Exception as e:
    print(f"\nError generating visualizations: {str(e)}")
    print("Please ensure all required packages are installed:")
    print("pip install matplotlib seaborn pandas")

# Generate comprehensive report
md_content = ["# ARC-Easy Vector Analysis Report (All Splits)\n"]

# 1. Dataset Overview
md_content.append("## 1. Dataset Overview\n")
md_content.append("| Split | Number of Examples |")
md_content.append("|-------|-------------------|")
md_content.append(f"| Train | {len(ds['train'])} |")
md_content.append(f"| Test | {len(ds['test'])} |")
md_content.append(f"| Validation | {len(ds['validation'])} |")
md_content.append(f"| Total | {len(all_examples)} |")

# 2. Task Type Distribution
md_content.append("\n## 2. Task Type Distribution\n")
md_content.append("| Task Type | Count |")
md_content.append("|-----------|-------|")
for task_type, count in results["task_type_counts"].items():
    md_content.append(f"| {task_type} | {count} |")

# 3. Descriptive Statistics
md_content.append("\n## 3. Descriptive Statistics\n")
md_content.append("| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |")
md_content.append("|-----------|-----|-----|------|---------|----|--------|----|")

for comp in components:
    if comp in stats:
        comp_stats = stats[comp]
        md_content.append(
            f"| {comp} | {comp_stats['min']:.3f} | {comp_stats['max']:.3f} | {comp_stats['mean']:.3f} | "
            f"{comp_stats['std']:.3f} | {comp_stats['q1']:.3f} | {comp_stats['median']:.3f} | {comp_stats['q3']:.3f} |"
        )

# 4. Summary Statistics
md_content.append("\n## 4. Summary Statistics\n")
md_content.append("| Total Questions | Successful | Failed | Success Rate |")
md_content.append("|----------------|------------|--------|--------------|")

total = len(results["vectors"]) + len(results["failed_questions"])
success_rate = len(results["vectors"]) / total * 100 if total > 0 else 0
md_content.append(
    f"| {total} | {len(results['vectors'])} | {len(results['failed_questions'])} | {success_rate:.1f}% |"
)

# 5. Key Findings
md_content.append("\n## 5. Key Findings\n")
md_content.append("### Vector Analysis\n")
md_content.append(
    "- Mean values show the average level of each dimension across all questions\n"
)
md_content.append("- Standard deviation indicates the variability in each dimension\n")
md_content.append("- Range (max-min) shows the spread of values for each dimension\n")

# 6. Visualizations
md_content.append("\n## 6. Visualizations\n")
md_content.append("### Component Distributions\n")
md_content.append(
    "![Component Distributions](visualizations/component_distributions.png)\n"
)
md_content.append(
    "This plot shows the distribution of values for each component, helping identify patterns and outliers. The KDE (Kernel Density Estimation) curve shows the probability density of the values.\n"
)

md_content.append("\n### Component Box Plots\n")
md_content.append("![Component Box Plots](visualizations/component_boxplots.png)\n")
md_content.append(
    "Box plots provide a clear view of the statistical distribution of each component, including median (center line), quartiles (box edges), and outliers (points beyond whiskers).\n"
)

md_content.append("\n### Component Correlations\n")
md_content.append("![Component Correlations](visualizations/correlation_heatmap.png)\n")
md_content.append(
    "This heatmap shows the correlation between different components, helping identify relationships between dimensions. Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).\n"
)

md_content.append("\n### Task Type Distribution\n")
md_content.append(
    "![Task Type Distribution](visualizations/task_type_distribution.png)\n"
)
md_content.append(
    "Pie chart showing the distribution of different task types in the dataset, with percentage labels for each category.\n"
)

md_content.append("\n### Component Scatter Matrix\n")
md_content.append(
    "![Component Scatter Matrix](visualizations/component_scatter_matrix.png)\n"
)
md_content.append(
    "This scatter plot matrix shows the relationships between all pairs of components, with histograms on the diagonal showing individual distributions.\n"
)

# Save to file
with open("arc_easy_vector_analysis_report.md", "w") as f:
    f.write("\n".join(md_content))

print("\nAnalysis report has been saved to 'arc_easy_vector_analysis_report.md'")
