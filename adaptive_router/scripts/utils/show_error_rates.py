#!/usr/bin/env python3
"""Display all model error rates across clusters."""

import json
from pathlib import Path

# Load error rates
profiles_file = (
    Path(__file__).parent.parent
    / "adaptive_router"
    / "data"
    / "unirouter"
    / "clusters"
    / "llm_profiles.json"
)

with open(profiles_file) as f:
    data = json.load(f)

print("=" * 100)
print("MODEL ERROR RATES ACROSS 20 CLUSTERS")
print("=" * 100)
print()

# Load costs for comparison
costs = {
    "anthropic:claude-3-5-haiku-20241022": 1.6,
    "anthropic:claude-sonnet-4-5-20250929": 6.0,
    "anthropic:claude-opus-4-1-20250805": 30.0,
    "openai:gpt-5-mini": 1.125,
    "openai:gpt-4.1-nano": 0.25,
    "gemini:gemini-2.5-flash-lite": 0.13125,
    "zai:glm-4.6": 0.5,  # estimated
    "openai:gpt-5-nano": 0.225,
    "openai:gpt-5-codex": 5.625,
    "anthropic:claude-haiku-4-5-20251001": 1.6,
}

for model_id, error_rates in sorted(data.items(), key=lambda x: sum(x[1]) / len(x[1])):
    avg_error = sum(error_rates) / len(error_rates)
    min_error = min(error_rates)
    max_error = max(error_rates)
    cost = costs.get(model_id, 0.0)

    print(f"Model: {model_id}")
    print(f"  Cost: ${cost:.3f} per 1M tokens")
    print(f"  Average Error Rate: {avg_error:.1%} (Accuracy: {(1-avg_error):.1%})")
    print(f"  Range: {min_error:.1%} - {max_error:.1%}")
    print(f"  Per-Cluster Errors:")

    for i, err in enumerate(error_rates):
        print(f"    Cluster {i:2d}: {err:5.1%}", end="")
        if (i + 1) % 5 == 0:
            print()
        else:
            print("  ", end="")
    if len(error_rates) % 5 != 0:
        print()
    print()

print("=" * 100)
print("QUALITY TIERS (by average accuracy)")
print("=" * 100)
print()

tiers = []
for model_id, error_rates in data.items():
    avg_error = sum(error_rates) / len(error_rates)
    avg_accuracy = 1 - avg_error
    cost = costs.get(model_id, 0.0)
    tiers.append((avg_accuracy, cost, model_id))

tiers.sort(reverse=True)

print("Tier 1: Premium (>95% accuracy)")
for acc, cost, model in tiers:
    if acc > 0.95:
        print(f"  {model:45s}  {acc:.1%} accuracy  ${cost:6.3f}/1M tokens")

print("\nTier 2: High Quality (90-95% accuracy)")
for acc, cost, model in tiers:
    if 0.90 <= acc <= 0.95:
        print(f"  {model:45s}  {acc:.1%} accuracy  ${cost:6.3f}/1M tokens")

print("\nTier 3: Good (85-90% accuracy)")
for acc, cost, model in tiers:
    if 0.85 <= acc < 0.90:
        print(f"  {model:45s}  {acc:.1%} accuracy  ${cost:6.3f}/1M tokens")

print("\nTier 4: Acceptable (<85% accuracy)")
for acc, cost, model in tiers:
    if acc < 0.85:
        print(f"  {model:45s}  {acc:.1%} accuracy  ${cost:6.3f}/1M tokens")
