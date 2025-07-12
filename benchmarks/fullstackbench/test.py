import json
from collections import Counter
from typing import Any

import numpy as np
import requests
from datasets import load_dataset

ds = load_dataset("ByteDance/FullStackBench", "en")

# Count difficulty levels
difficulty_counts = Counter(example["labels"]["difficulty"] for example in ds["test"])
print("\nDifficulty level distribution:")
for difficulty, count in difficulty_counts.items():
    print(f"{difficulty}: {count} examples")

# Get examples by difficulty
easy_examples = [ex for ex in ds["test"] if ex["labels"]["difficulty"] == "easy"]
hard_examples = [ex for ex in ds["test"] if ex["labels"]["difficulty"] == "hard"]

print("\nTotal examples by difficulty:")
print(f"Easy: {len(easy_examples)}")
print(f"Hard: {len(hard_examples)}")


# Function to process examples for a given difficulty level
def process_difficulty_level(examples, difficulty: str) -> dict[str, Any]:
    API_URL = "http://localhost:8000/predict"
    vectors = []
    failed_questions: list[int] = []
    task_type_counts: Counter = Counter()

    print(f"\nAnalyzing API responses for {difficulty} questions...")
    for i, example in enumerate(examples):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing question {i}/{len(examples)}")

        request_data = {"prompt": example["content"]}
        try:
            response = requests.post(API_URL, json=request_data)
            result = response.json()

            if "weighted_vector" not in result:
                if len(failed_questions) < 5:  # Only show details for first 5 failures
                    print(f"\nFailed question {i}:")
                    print(f"Question: {example['content']}")
                    print(f"Full API Response: {json.dumps(result, indent=2)}")
                failed_questions.append(i)
                continue

            vectors.append(result["weighted_vector"])
            if "task_type" in result:
                task_type_counts[result["task_type"]] += 1
        except Exception as e:
            if len(failed_questions) < 5:  # Only show details for first 5 failures
                print(f"\nError processing question {i}:")
                if hasattr(e, "response"):
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response text: {e.response.text}")
            failed_questions.append(i)

    print(f"\nTotal successful vectors for {difficulty}: {len(vectors)}")
    print(f"Total failed questions for {difficulty}: {len(failed_questions)}")
    print(
        f"Failed question indices: {failed_questions[:10]}..."
    )  # Show first 10 failed indices

    print(f"\nTask type distribution for {difficulty}:")
    for task_type, count in task_type_counts.items():
        print(f"{task_type}: {count}")

    return {
        "vectors": vectors,
        "failed_questions": failed_questions,
        "task_type_counts": task_type_counts,
    }


# Process Easy and Hard difficulty levels
results = {
    "easy": process_difficulty_level(easy_examples, "easy"),
    "hard": process_difficulty_level(hard_examples, "hard"),
}


# Function to calculate statistics for vectors
def calculate_vector_stats(
    vectors: list[list[float]], components: list[str]
) -> dict[str, dict[str, float]]:
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

# Calculate statistics for each difficulty level
difficulty_stats = {
    difficulty: calculate_vector_stats(data["vectors"], components)
    for difficulty, data in results.items()
}

# Generate comprehensive report
md_content = ["# FullStackBench Vector Analysis Report - Easy vs Hard Comparison\n"]

# 1. Task Type Distribution
md_content.append("## 1. Task Type Distribution\n")
for difficulty, data in results.items():
    md_content.append(f"\n### {difficulty.capitalize()} Questions\n")
    md_content.append("| Task Type | Count |")
    md_content.append("|-----------|-------|")
    for task_type, count in data["task_type_counts"].items():
        md_content.append(f"| {task_type} | {count} |")

# 2. Descriptive Statistics
md_content.append("\n## 2. Descriptive Statistics per Difficulty Level\n")
for difficulty, level_stats in difficulty_stats.items():
    md_content.append(f"\n### {difficulty.capitalize()} Questions\n")
    md_content.append("| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |")
    md_content.append("|-----------|-----|-----|------|---------|----|--------|----|")

    for comp in components:
        if comp in level_stats:
            comp_stats = level_stats[comp]
            md_content.append(
                f"| {comp} | {comp_stats['min']:.3f} | {comp_stats['max']:.3f} | {comp_stats['mean']:.3f} | "
                f"{comp_stats['std']:.3f} | {comp_stats['q1']:.3f} | {comp_stats['median']:.3f} | {comp_stats['q3']:.3f} |"
            )

# 3. Comparison Statistics
md_content.append("\n## 3. Easy vs Hard Comparison\n")

# Mean Values Comparison
md_content.append("### Mean Values Comparison\n")
md_content.append("| Dimension | Easy | Hard | Difference | % Difference |")
md_content.append("|-----------|------|------|------------|--------------|")

for comp in components:
    easy_mean = difficulty_stats["easy"][comp]["mean"]
    hard_mean = difficulty_stats["hard"][comp]["mean"]
    diff = hard_mean - easy_mean
    pct_diff = (diff / easy_mean * 100) if easy_mean != 0 else 0
    md_content.append(
        f"| {comp} | {easy_mean:.3f} | {hard_mean:.3f} | {diff:+.3f} | {pct_diff:+.1f}% |"
    )

# Distribution Comparison
md_content.append("\n### Distribution Comparison\n")
md_content.append("| Dimension | Easy Std Dev | Hard Std Dev | Difference |")
md_content.append("|-----------|-------------|-------------|------------|")

for comp in components:
    easy_std = difficulty_stats["easy"][comp]["std"]
    hard_std = difficulty_stats["hard"][comp]["std"]
    diff = hard_std - easy_std
    md_content.append(f"| {comp} | {easy_std:.3f} | {hard_std:.3f} | {diff:+.3f} |")

# Range Comparison
md_content.append("\n### Range Comparison\n")
md_content.append("| Dimension | Easy Range | Hard Range | Difference |")
md_content.append("|-----------|------------|------------|------------|")

for comp in components:
    easy_range = (
        difficulty_stats["easy"][comp]["max"] - difficulty_stats["easy"][comp]["min"]
    )
    hard_range = (
        difficulty_stats["hard"][comp]["max"] - difficulty_stats["hard"][comp]["min"]
    )
    diff = hard_range - easy_range
    md_content.append(f"| {comp} | {easy_range:.3f} | {hard_range:.3f} | {diff:+.3f} |")

# 4. Summary Statistics
md_content.append("\n## 4. Summary Statistics\n")
md_content.append(
    "| Difficulty | Total Questions | Successful | Failed | Success Rate |"
)
md_content.append(
    "|------------|----------------|------------|--------|--------------|"
)

for difficulty, data in results.items():
    total = len(data["vectors"]) + len(data["failed_questions"])
    success_rate = len(data["vectors"]) / total * 100 if total > 0 else 0
    md_content.append(
        f"| {difficulty.capitalize()} | {total} | {len(data['vectors'])} | "
        f"{len(data['failed_questions'])} | {success_rate:.1f}% |"
    )

# 5. Key Findings
md_content.append("\n## 5. Key Findings\n")
md_content.append("### Mean Value Differences\n")
md_content.append("- Positive differences indicate higher values in Hard questions\n")
md_content.append("- Negative differences indicate higher values in Easy questions\n")
md_content.append(
    "- Percentage differences show relative change between difficulty levels\n"
)

md_content.append("\n### Distribution Differences\n")
md_content.append(
    "- Larger standard deviation in Hard questions indicates more variability\n"
)
md_content.append(
    "- Range differences show the spread of values in each difficulty level\n"
)

# Save to file
with open("fullstackbench_vector_analysis_report.md", "w") as f:
    f.write("\n".join(md_content))

print("\nAnalysis report has been saved to 'fullstackbench_vector_analysis_report.md'")
