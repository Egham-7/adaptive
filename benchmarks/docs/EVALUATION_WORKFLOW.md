# HumanEval Evaluation Workflow

This document provides a comprehensive guide to the HumanEval benchmark evaluation system, covering architecture, usage, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Quick Start](#quick-start)
4. [Detailed Workflow](#detailed-workflow)
5. [Metrics Explained](#metrics-explained)
6. [Configuration](#configuration)
7. [Results & Reporting](#results--reporting)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The HumanEval Evaluation System is a comprehensive benchmarking framework built on top of [DeepEval](https://github.com/confident-ai/deepeval) that evaluates large language models on coding tasks from the HumanEval dataset.

### Supported Models

- **Claude 4.5 Sonnet** - Anthropic's latest model via the Anthropic SDK
- **GLM-4.6** - ZhipuAI's model via direct API integration
- **Adaptive Router** - Intelligent routing between multiple models with cost optimization

### Key Features

- **DeepEval Integration**: Uses DeepEval's battle-tested HumanEval implementation
- **Comprehensive Cost Tracking**: Tracks tokens, costs, and latency per sample
- **Multi-format Reports**: JSON, CSV, and Markdown outputs
- **Model Comparison**: Compare performance and cost efficiency across models
- **Pass@k Metrics**: Standard pass@10 metric with configurable k values

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / Entry Point                         │
│                  (run_benchmark.py)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Layer                        │
│           (config.py - BenchmarkSettings)                    │
│  • Load environment variables                                │
│  • Validate API keys                                         │
│  • Set benchmark parameters                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Adapters                            │
│      (ClaudeForDeepEval, GLMForDeepEval,                    │
│           AdaptiveForDeepEval)                               │
│  • Implement DeepEval's model interface                      │
│  • Add cost tracking layer                                   │
│  • Parse API responses                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DeepEval Benchmark                         │
│               (HumanEval Framework)                          │
│  • Load 164 coding tasks                                     │
│  • Generate n samples per task                               │
│  • Run unit tests on generated code                          │
│  • Calculate pass@k scores                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Result Processing                          │
│        (ResultTracker, Reporting Utils)                      │
│  • Aggregate task metrics                                    │
│  • Calculate overall statistics                              │
│  • Generate reports (JSON, CSV, Markdown)                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request
    ↓
Configuration Loading (.env.local)
    ↓
Model Initialization (SDK clients)
    ↓
DeepEval Benchmark Creation (164 tasks)
    ↓
FOR EACH Task (1-164):
    ↓
    FOR EACH Sample (1-200):
        ↓
        API Call → Generate Code
        ↓
        Extract Metrics (tokens, cost, latency)
        ↓
        Store in TaskMetrics
    END
    ↓
    DeepEval runs unit tests on all 200 samples
    ↓
    Calculate pass@k for this task
END
    ↓
Aggregate all TaskMetrics → BenchmarkRun
    ↓
Save Results (JSON, CSV, Markdown)
    ↓
Display Summary to User
```

---

## Quick Start

### 1. Installation

```bash
cd /path/to/benchmarks
uv sync  # or pip install -e .
```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```bash
# Required: Add at least one model's API key
ANTHROPIC_API_KEY=your_claude_key_here
GLM_API_KEY=your_glm_key_here
ADAPTIVE_API_KEY=your_adaptive_key_here

# Optional: Customize benchmark parameters
HUMANEVAL_N_SAMPLES=200    # Samples per task
HUMANEVAL_K_VALUE=10       # Pass@k metric
```

### 3. Run a Quick Test

Test with a single task (2-3 minutes, ~$0.50):

```bash
uv run python humaneval/run_benchmark.py claude --quick
```

### 4. Run Full Benchmark

Run all 164 tasks (2-4 hours, ~$40-60):

```bash
uv run python humaneval/run_benchmark.py claude --full
```

### 5. Compare Results

```bash
uv run python humaneval/run_benchmark.py compare
```

---

## Detailed Workflow

### Phase 1: Setup & Configuration

**What happens:**
1. Environment variables loaded from `.env` and `.env.local`
2. `BenchmarkSettings` object created with all configurations
3. API keys validated for the requested model
4. Results directory created at `results/{model_name}/`

**Key files:**
- `humaneval/config.py` - Configuration classes
- `.env.local` - Your API keys and settings

**Configuration options:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HUMANEVAL_N_SAMPLES` | 200 | Samples to generate per task |
| `HUMANEVAL_K_VALUE` | 10 | K value for pass@k metric |
| `DEEPEVAL_RESULTS_FOLDER` | results | Output directory |
| `ANTHROPIC_API_KEY` | - | Claude API key |
| `GLM_API_KEY` | - | GLM API key |
| `ADAPTIVE_API_KEY` | - | Adaptive API key |

### Phase 2: Model Initialization

**What happens:**
1. Model-specific SDK client created (Anthropic, OpenAI-compatible, etc.)
2. DeepEval adapter class wraps the client
3. `ResultTracker` initialized to collect metrics

**Adapter responsibilities:**
- Implement `generate_samples(prompt, n, temperature)` method
- Parse API responses to extract code and metrics
- Track costs, tokens, and latency for each sample
- Store metrics in `TaskMetrics` objects

### Phase 3: Benchmark Execution

**What happens:**
1. DeepEval `HumanEval` benchmark object created
2. For each of 164 tasks:
   - Model generates `n` code samples (default: 200)
   - Each sample tracked: code, tokens, cost, latency
   - DeepEval runs unit tests on all samples
   - Pass/fail results recorded
3. DeepEval calculates overall pass@k score

**Timeline (full benchmark):**
- **Claude 4.5**: ~2-3 hours, ~$40-50
- **GLM-4.6**: ~3-4 hours, ~$30-40
- **Adaptive**: ~2-3 hours, ~$25-35 (cost-optimized)

**Sample generation example:**

```python
# For task "HumanEval/0: Sort Numbers"
for i in range(200):  # Generate 200 samples
    start_time = time.time()

    # Call model API
    response = model_api.generate(
        prompt="Write a function to sort numbers...",
        temperature=0.8,
        max_tokens=1024
    )

    # Extract metrics
    code = response.content
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(input_tokens, output_tokens)
    latency = time.time() - start_time

    # Store
    task_metrics.add_sample(code, cost, latency, ...)
```

### Phase 4: Results Aggregation

**What happens:**
1. All `TaskMetrics` collected from model adapter
2. Aggregated into `BenchmarkRun` object:
   - Total cost across all 164 tasks
   - Total tokens (input + output)
   - Average latency
   - Success rate
   - Pass@k score from DeepEval
3. Model-specific stats (e.g., Adaptive's model selection)

**Example aggregation:**

```python
BenchmarkRun(
    model_name="claude-sonnet-4-5",
    overall_score=0.847,           # pass@10 score
    pass_at_k={"k": 10, "score": 0.847},
    total_tasks=164,
    total_samples=32800,           # 164 × 200
    total_input_tokens=12456789,
    total_output_tokens=3234567,
    total_cost_usd=45.67,
    avg_latency_seconds=2.3,
    success_rate=0.998,            # 99.8% API success rate
    per_task_results=[...],        # 164 task details
)
```

### Phase 5: Output Generation

**What happens:**
Four types of output files generated in `results/{model_name}/`:

1. **Detailed JSON** (`{model}_{timestamp}.json`)
   - Full results with per-sample costs
   - ~50-100MB file size
   - For deep analysis

2. **Summary JSON** (`{model}_{timestamp}_summary.json`)
   - Aggregated metrics without per-sample arrays
   - ~1-2MB file size
   - For quick loading

3. **CSV Export** (`{model}_{timestamp}.csv`)
   - Task-level summary
   - Easy to open in Excel/spreadsheet tools
   - Columns: task_id, cost, tokens, latency, etc.

4. **Markdown Report** (`{model}_{timestamp}_report.md`)
   - Human-readable summary
   - Cost breakdown
   - Top expensive tasks
   - Failed tasks analysis

**Example output structure:**

```
results/
├── claude/
│   ├── claude-sonnet-4-5_20251112_143021.json
│   ├── claude-sonnet-4-5_20251112_143021_summary.json
│   ├── claude-sonnet-4-5_20251112_143021.csv
│   └── claude-sonnet-4-5_20251112_143021_report.md
├── glm/
│   └── (similar files)
├── adaptive/
│   └── (similar files)
└── comparisons/
    └── comparison_20251112_150000.md
```

---

## Metrics Explained

### Pass@k Score

**Definition**: Probability that at least 1 of k generated samples passes all unit tests.

**Formula**: `pass@k = E[1 - C(n-c, k) / C(n, k)]`

Where:
- `n` = total samples generated (default: 200)
- `c` = number of correct samples
- `k` = samples considered (default: 10)

**Interpretation:**
- `pass@10 = 0.847` → 84.7% chance that at least 1 of 10 samples is correct
- Higher is better (0.0 - 1.0 range)
- Industry standard for code generation benchmarks

**Why pass@10?**
- Balances between evaluation thoroughness and practical use
- Reflects real-world scenario: developer reviews top 10 suggestions
- Standard in HumanEval paper and research

### Cost Metrics

**Per-Sample Metrics:**
- `input_tokens`: Tokens in the prompt
- `output_tokens`: Tokens in the generated code
- `cost_usd`: Cost for this API call
- `latency_seconds`: Time to generate

**Aggregated Metrics:**
- `total_cost_usd`: Sum across all 32,800 samples
- `avg_cost_per_task`: Total cost ÷ 164 tasks
- `avg_cost_per_sample`: Total cost ÷ 32,800 samples

**Cost tracking methods:**
1. **Response-based** (preferred): Extract from API response
2. **Calculated** (fallback): Use pricing calculator with token counts

### Performance Metrics

**Success Rate:**
- Percentage of API calls that succeeded
- Failed calls tracked separately with error messages

**Average Latency:**
- Weighted by successful samples
- Excludes failed/timeout samples
- Measured in seconds

**Model Selection (Adaptive only):**
- Which model was chosen for each sample
- Distribution across available models
- Helps evaluate routing decisions

---

## Configuration

### Environment Variables

Create `.env.local` with your settings:

```bash
# ============================================
# Model API Keys (required for each model)
# ============================================
ANTHROPIC_API_KEY=sk-ant-...
GLM_API_KEY=...
ADAPTIVE_API_KEY=...

# ============================================
# Benchmark Parameters
# ============================================

# Number of code samples to generate per task
# Higher = more accurate pass@k, but more expensive
# Default: 200 (standard for HumanEval)
HUMANEVAL_N_SAMPLES=200

# Pass@k metric - how many samples to consider
# Common values: 1, 5, 10, 100
# Default: 10
HUMANEVAL_K_VALUE=10

# Temperature for code generation
# Higher = more creative but less consistent
# Range: 0.0 - 2.0
# Default: 0.8
HUMANEVAL_TEMPERATURE=0.8

# Max tokens per code generation
# Default: 1024
HUMANEVAL_MAX_TOKENS=1024

# ============================================
# Model-Specific Settings
# ============================================

# Claude model to use
# Options: claude-sonnet-4-5, claude-opus-4, etc.
CLAUDE_MODEL=claude-sonnet-4-5

# GLM model and endpoint
GLM_MODEL=glm-4.6
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/

# Adaptive routing settings
ADAPTIVE_API_BASE=https://api.llmadaptive.uk/v1/
ADAPTIVE_MODELS=anthropic:claude-sonnet-4-5-20250929,z-ai:glm-4.6
ADAPTIVE_COST_BIAS=0.5  # 0=speed, 1=cost

# ============================================
# Execution Settings
# ============================================

# Enable parallel execution (currently not implemented)
BENCHMARK_PARALLEL=false

# Results output folder
DEEPEVAL_RESULTS_FOLDER=results

# Retry configuration
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_SECONDS=1.0
RETRY_EXP_BASE=2.0
RETRY_CAP_SECONDS=10.0
```

### Programmatic Configuration

```python
from humaneval.config import BenchmarkSettings

# Load from environment
settings = BenchmarkSettings()

# Print summary
settings.print_summary()

# Validate model config
settings.validate_model("claude")

# Access configurations
print(f"Samples per task: {settings.humaneval.n_samples}")
print(f"Claude model: {settings.claude.model_name}")
```

---

## Results & Reporting

### Reading JSON Results

```python
from humaneval.utils import ResultTracker

# Load results
run = ResultTracker.load_from_json("results/claude/claude-sonnet-4-5_20251112_143021.json")

# Access metrics
print(f"Pass@{run.pass_at_k['k']}: {run.overall_score:.4f}")
print(f"Total cost: ${run.total_cost_usd:.2f}")
print(f"Average cost per task: ${run.avg_cost_per_task:.4f}")
print(f"Average latency: {run.avg_latency_seconds:.2f}s")

# Iterate over task results
for task in run.per_task_results:
    print(f"{task['task_id']}: ${task['total_cost_usd']:.4f}")
```

### Comparison Reports

Generate a comparison across models:

```bash
uv run python humaneval/run_benchmark.py compare
```

Example comparison output:

```markdown
# Model Comparison Report

## Performance Ranking

| Rank | Model | Pass@10 | Cost | Cost Efficiency |
|------|-------|---------|------|-----------------|
| 1 | Claude 4.5 | 0.847 | $45.67 | 0.0185 |
| 2 | GLM-4.6 | 0.823 | $32.45 | 0.0254 |
| 3 | Adaptive | 0.841 | $28.12 | 0.0299 |

## Cost Analysis

- **Most Expensive**: Claude 4.5 at $45.67
- **Most Affordable**: Adaptive at $28.12 (38% savings vs Claude)
- **Best Value**: Adaptive with 0.0299 score per dollar

## Token Usage

- Claude: 15.7M tokens (12.5M in, 3.2M out)
- GLM: 18.2M tokens (14.1M in, 4.1M out)
- Adaptive: 14.8M tokens (11.9M in, 2.9M out)
```

### Markdown Reports

Each benchmark generates a detailed markdown report:

```markdown
# HumanEval Benchmark Results: Claude 4.5

**Date**: 2025-11-12 14:30:21
**Model**: claude-sonnet-4-5

## Overall Performance

- **Pass@10 Score**: 0.847 (84.7%)
- **Total Tasks**: 164
- **Total Samples**: 32,800 (200 per task)
- **Success Rate**: 99.8%

## Cost Breakdown

- **Total Cost**: $45.67
- **Average per Task**: $0.278
- **Average per Sample**: $0.0014
- **Token Usage**: 15,723,456 total
  - Input: 12,534,567 tokens
  - Output: 3,188,889 tokens

## Performance

- **Average Latency**: 2.31 seconds per sample
- **Total Runtime**: ~2.5 hours

## Most Expensive Tasks

1. HumanEval/143: $0.89 (complex algorithm)
2. HumanEval/089: $0.76 (requires long context)
3. HumanEval/127: $0.71 (multi-step problem)
...

## Failed Tasks

- HumanEval/045: 0/200 samples passed (pass@10 = 0.000)
- HumanEval/132: 3/200 samples passed (pass@10 = 0.142)
...
```

---

## Best Practices

### 1. Start with Quick Tests

Always run `--quick` first to verify configuration:

```bash
# Test each model before full run
uv run python humaneval/run_benchmark.py claude --quick
uv run python humaneval/run_benchmark.py glm --quick
uv run python humaneval/run_benchmark.py adaptive --quick
```

### 2. Monitor Costs

Full benchmarks are expensive. Budget estimates:

| Model | n=50 (quick) | n=200 (full) |
|-------|--------------|--------------|
| Claude 4.5 | ~$1.50 | ~$45-50 |
| GLM-4.6 | ~$1.00 | ~$30-40 |
| Adaptive | ~$0.80 | ~$25-35 |

### 3. Use Appropriate n Values

- **Quick testing**: n=50 (1 task, $0.50-1.50, 2-3 min)
- **Development**: n=100 (10 tasks, $5-10, 20-30 min)
- **Production**: n=200 (164 tasks, $25-50, 2-4 hours)

### 4. Version Control Results

Add results to `.gitignore` by default. Store important benchmark runs separately:

```bash
# Save important results
mkdir -p benchmarks_archive/2024-11-12
cp -r results/ benchmarks_archive/2024-11-12/
```

### 5. Compare Apples to Apples

When comparing models, use:
- Same `n` value
- Same `k` value
- Same HumanEval tasks
- Similar time periods (avoid API changes)

---

## Troubleshooting

### Issue: API Key Not Found

```
ValueError: ANTHROPIC_API_KEY not found in environment
```

**Solution:**
1. Check `.env.local` exists and has the key
2. Verify key format (no quotes, no spaces)
3. Try `source .env.local` if using bash directly

### Issue: Rate Limiting

```
RateLimitError: Rate limit exceeded
```

**Solution:**
1. Wait and retry (exponential backoff implemented)
2. Reduce `HUMANEVAL_N_SAMPLES` temporarily
3. Contact provider to increase rate limits

### Issue: Out of Memory

```
MemoryError: Cannot allocate memory
```

**Solution:**
1. Run one model at a time (not `--all`)
2. Close other applications
3. Use `--quick` to verify, then run full benchmark
4. Process results in batches

### Issue: DeepEval Import Error

```
ModuleNotFoundError: No module named 'deepeval'
```

**Solution:**
```bash
uv sync  # Re-sync dependencies
# or
pip install -e .
```

### Issue: Results Not Saving

Check permissions and disk space:

```bash
# Check disk space
df -h .

# Check permissions
ls -la results/

# Create directory if needed
mkdir -p results/claude
chmod 755 results/claude
```

### Issue: Inconsistent Scores

Pass@k has some variance. For consistent results:
- Use the same `n` and `k` values
- Run multiple times and average
- Use `temperature=0.0` for deterministic output (may reduce creativity)

---

## Advanced Usage

### Custom Task Subset

Test specific tasks:

```python
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask

benchmark = HumanEval(
    tasks=[
        HumanEvalTask.SORT_NUMBERS,
        HumanEvalTask.HAS_CLOSE_ELEMENTS,
        HumanEvalTask.STRLEN,
    ],
    n=100
)
```

### Pytest Integration

Run via pytest:

```bash
# Run specific test
pytest humaneval/tests/test_claude.py::TestClaudeHumanEval::test_claude_humaneval_quick -v

# Run all integration tests
pytest -m integration

# Run with output
pytest -s -v
```

### Direct DeepEval CLI

```bash
# Use DeepEval's CLI directly
deepeval test run humaneval/tests/test_claude.py -v
```

---

## Summary

This evaluation workflow provides:

✅ **Standardized benchmarking** using the well-established HumanEval dataset
✅ **Comprehensive cost tracking** down to the per-sample level
✅ **Multiple output formats** for different analysis needs
✅ **Easy comparison** across different models
✅ **Configurable parameters** for different use cases

The system is production-ready for evaluating and comparing LLMs on coding tasks, with full transparency into costs and performance characteristics.

For questions or issues, see the main [README](../README.md) or open an issue on GitHub.
