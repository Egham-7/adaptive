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
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
all_examples = concatenate_datasets([ds["train"], ds["test"], ds["validation"]])

# Take only first 100 examples for testing
all_examples = all_examples.select(range(100))

print(f"\nProcessing first 100 examples from the dataset")

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

        if "complexity_score" not in result:
            return {"failed": True, "index": i, "prompt": prompt, "result": result}

        return {
            "failed": False,
            "complexity_score": result["complexity_score"],
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
    complexity_scores = []
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
                complexity_scores.append(result["complexity_score"])
                if result["task_type"]:
                    task_type_counts[result["task_type"]] += 1

    print(f"\nTotal successful complexity scores: {len(complexity_scores)}")
    print(f"Total failed questions: {len(failed_questions)}")
    print(
        f"Failed question indices: {failed_questions[:10]}..."
    )  # Show first 10 failed indices

    print(f"\nTask type distribution:")
    for task_type, count in task_type_counts.items():
        print(f"{task_type}: {count}")

    return {
        "complexity_scores": complexity_scores,
        "failed_questions": failed_questions,
        "task_type_counts": task_type_counts,
    }


# Process all examples
results = process_examples(all_examples)


# Function to calculate statistics for complexity scores
def calculate_complexity_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {}

    scores_array = np.array(scores)
    return {
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array)),
        "q1": float(np.percentile(scores_array, 25)),
        "median": float(np.percentile(scores_array, 50)),
        "q3": float(np.percentile(scores_array, 75)),
    }


# Calculate statistics
stats = calculate_complexity_stats(results["complexity_scores"])

# Create visualizations directory if it doesn't exist
visualizations_dir = Path("visualizations")
visualizations_dir.mkdir(exist_ok=True)

try:
    # 1. Distribution plot for complexity scores
    plt.figure(figsize=(10, 6))
    sns.histplot(results["complexity_scores"], kde=True)
    plt.title("Complexity Score Distribution")
    plt.xlabel("Complexity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "complexity_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Box plot for complexity scores
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=results["complexity_scores"])
    plt.title("Complexity Score Distribution")
    plt.ylabel("Complexity Score")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "complexity_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Task type distribution pie chart
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

    # 4. Complexity scores by task type
    plt.figure(figsize=(12, 6))
    task_complexity_data = []
    for task_type, count in results["task_type_counts"].items():
        task_scores = [
            score
            for i, score in enumerate(results["complexity_scores"])
            if results["task_type_counts"][task_type] > 0
        ]
        task_complexity_data.extend([(task_type, score) for score in task_scores])

    df = pd.DataFrame(task_complexity_data, columns=["Task Type", "Complexity Score"])
    sns.boxplot(x="Task Type", y="Complexity Score", data=df)
    plt.xticks(rotation=45)
    plt.title("Complexity Scores by Task Type")
    plt.tight_layout()
    plt.savefig(
        visualizations_dir / "complexity_by_task_type.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        "\nAll visualizations have been successfully saved to the 'visualizations' directory"
    )
    print("Generated plots:")
    print("1. complexity_distribution.png - Distribution of complexity scores")
    print("2. complexity_boxplot.png - Box plot of complexity scores")
    print("3. task_type_distribution.png - Distribution of task types")
    print("4. complexity_by_task_type.png - Complexity scores by task type")

except Exception as e:
    print(f"\nError generating visualizations: {str(e)}")
    print("Please ensure all required packages are installed:")
    print("pip install matplotlib seaborn pandas")

# Generate comprehensive report
md_content = ["# ARC-Easy Complexity Score Analysis Report\n"]

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

# 3. Complexity Score Statistics
md_content.append("\n## 3. Complexity Score Statistics\n")
md_content.append("| Statistic | Value |")
md_content.append("|-----------|-------|")
for stat, value in stats.items():
    md_content.append(f"| {stat.capitalize()} | {value:.3f} |")

# 4. Summary Statistics
md_content.append("\n## 4. Summary Statistics\n")
md_content.append("| Total Questions | Successful | Failed | Success Rate |")
md_content.append("|----------------|------------|--------|--------------|")

total = len(results["complexity_scores"]) + len(results["failed_questions"])
success_rate = len(results["complexity_scores"]) / total * 100 if total > 0 else 0
md_content.append(
    f"| {total} | {len(results['complexity_scores'])} | {len(results['failed_questions'])} | {success_rate:.1f}% |"
)

# 5. Key Findings
md_content.append("\n## 5. Key Findings\n")
md_content.append("### Complexity Score Analysis\n")
md_content.append(
    "- Mean complexity score shows the average difficulty level across all questions\n"
)
md_content.append("- Standard deviation indicates the variability in complexity\n")
md_content.append("- Range (max-min) shows the spread of complexity scores\n")

# 6. Visualizations
md_content.append("\n## 6. Visualizations\n")
md_content.append("### Complexity Score Distribution\n")
md_content.append(
    "![Complexity Distribution](visualizations/complexity_distribution.png)\n"
)
md_content.append(
    "This plot shows the distribution of complexity scores, helping identify patterns and outliers. The KDE (Kernel Density Estimation) curve shows the probability density of the scores.\n"
)

md_content.append("\n### Complexity Score Box Plot\n")
md_content.append("![Complexity Box Plot](visualizations/complexity_boxplot.png)\n")
md_content.append(
    "Box plot provides a clear view of the statistical distribution of complexity scores, including median (center line), quartiles (box edges), and outliers (points beyond whiskers).\n"
)

md_content.append("\n### Task Type Distribution\n")
md_content.append(
    "![Task Type Distribution](visualizations/task_type_distribution.png)\n"
)
md_content.append(
    "Pie chart showing the distribution of different task types in the dataset, with percentage labels for each category.\n"
)

md_content.append("\n### Complexity by Task Type\n")
md_content.append(
    "![Complexity by Task Type](visualizations/complexity_by_task_type.png)\n"
)
md_content.append(
    "This box plot shows how complexity scores vary across different task types, helping identify which types of tasks tend to be more complex.\n"
)

# Save to file
with open("arc_easy_complexity_analysis_report.md", "w") as f:
    f.write("\n".join(md_content))

print("\nAnalysis report has been saved to 'arc_easy_complexity_analysis_report.md'")
