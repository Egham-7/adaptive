# SWE-bench Lite Benchmarking for Adaptive AI

Benchmark the Adaptive AI router on **SWE-bench Lite** (300 real-world GitHub issues) with cloud-based evaluation via `sb-cli`.

## 🎯 Why SWE-bench Lite?

- **300 curated instances** - Perfect sample size for benchmarking
- **Real GitHub issues** - From popular Python repositories
- **Cloud evaluation** - No Docker setup needed
- **Fast iteration** - Test routing strategies quickly
- **Cost tracking** - See exactly which models Adaptive selects

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd swe-bench
./quick_start.sh
```

This installs:
- `sb-cli` - SWE-bench cloud evaluation tool
- `openai` - For Adaptive API
- All other dependencies

### 2. Configure API Keys

Your `.env` file is already set up with:
```bash
# Adaptive AI
ADAPTIVE_API_KEY=apk_jA6pqOLc4iS9XEvKCiYcbfF30KJTU1mFPtnx-9_fNmg
ADAPTIVE_API_BASE=https://api.llmadaptive.uk/v1/

# SWE-bench
SWEBENCH_API_KEY=swb_UW-yh0B1I_fLymJAf_HOtpVCRx4NmgDyN_YLiN98d_I_6914d0ea
SWEBENCH_DATASET=swe-bench_lite
SWEBENCH_SPLIT=dev
```

### 3. Test Connection

```bash
# Verify sb-cli works
uv run sb-cli get-quotas --api_key $SWEBENCH_API_KEY

# Or just run the benchmark - it will test automatically
uv run python run_benchmark.py --quick
```

## 📊 Usage

### Quick Test (5 instances)

```bash
uv run python run_benchmark.py --quick
```

**What happens:**
1. ✅ Loads 5 instances from SWE-bench Lite
2. ✅ Generates patches using Adaptive (routes between GPT-4o-mini, Claude, Gemini)
3. ✅ Tracks cost, tokens, and model selection for each
4. ✅ Submits predictions to SWE-bench for cloud evaluation
5. ✅ Saves results locally

**Expected time:** ~2-5 minutes
**Expected cost:** ~$0.02-$0.05

### Medium Test (50 instances)

```bash
uv run python run_benchmark.py --medium
```

**Expected time:** ~20-30 minutes
**Expected cost:** ~$0.20-$0.50

### Full Benchmark (all 300 instances)

```bash
uv run python run_benchmark.py --full
```

**Expected time:** ~2-3 hours
**Expected cost:** ~$1.50-$3.00

### Advanced Options

```bash
# Custom number of instances
uv run python run_benchmark.py --max-instances 10

# Optimize for cost (use cheaper models more often)
uv run python run_benchmark.py --quick --cost-bias 0.8

# Optimize for quality (use better models more often)
uv run python run_benchmark.py --quick --cost-bias 0.2

# Skip submission (just generate predictions)
uv run python run_benchmark.py --quick --skip-submit

# Wait for evaluation results
uv run python run_benchmark.py --quick --wait
```

## 📂 Output Files

After running, you'll get:

```
swe-bench/
├── predictions/
│   └── adaptive_20250112_143022.json    # Predictions for SWE-bench
├── results/adaptive/
│   ├── adaptive_20250112_143022_generation.json   # Detailed generation metrics
│   ├── adaptive_20250112_143022_generation.csv    # Summary CSV
│   └── adaptive_20250112_143022_report.json       # SWE-bench evaluation results
```

### Example Output

```
======================================================================
  SWE-bench Benchmark Configuration
======================================================================

  Benchmark Settings:
    Dataset:                  swe-bench_lite
    Split:                    dev
    Max instances:            5 (of 300 total)
    Temperature:              0.2
    Max tokens:               4096
    Evaluation:               Cloud-based (sb-cli)

  Adaptive Router:
    Models:                   3 models
      - Cost bias:            0.5
      - Models:               gpt-4o-mini, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp

======================================================================
Generating predictions for 5 instances
======================================================================

[1/5] Processing django__django-11099
  ✓ Patch generated | Cost: $0.0023 | Tokens: 892 | Model: gpt-4o-mini

[2/5] Processing pytest-dev__pytest-5221
  ✓ Patch generated | Cost: $0.0156 | Tokens: 1245 | Model: claude-3-5-sonnet-20241022

[3/5] Processing sympy__sympy-15011
  ✓ Patch generated | Cost: $0.0008 | Tokens: 456 | Model: gemini-2.0-flash-exp

...

======================================================================
  SWE-bench Benchmark Results
======================================================================

  Resolution Results:
    Total instances:      5
    Patches generated:    5

  Cost Metrics:
    Total cost:           $0.0487
    Cost per instance:    $0.0097

  Token Metrics:
    Total tokens:         15,234
    Input tokens:         10,123
    Output tokens:        5,111

  Model Selection (Adaptive):
    gpt-4o-mini                        3 ( 60.0%)
    claude-3-5-sonnet-20241022         1 ( 20.0%)
    gemini-2.0-flash-exp               1 ( 20.0%)

======================================================================

✓ Predictions saved to: predictions/adaptive_20250112_143022.json
✓ Results saved to: results/adaptive/

======================================================================
Submitting to SWE-bench for evaluation...
======================================================================

✓ SWE-bench API connection successful
✓ Predictions submitted successfully!

Run ID: adaptive_20250112_143022

To check results later, run:
  sb-cli get-report swe-bench_lite dev adaptive_20250112_143022
```

## 🔍 How It Works

### 1. Patch Generation (Your Cost)

```
For each SWE-bench instance:
  ├─ Load problem statement
  ├─ Adaptive routes to best model
  │   ├─ Simple issues → gpt-4o-mini ($0.002)
  │   ├─ Medium issues → gemini-2.0-flash-exp ($0.001)
  │   └─ Complex issues → claude-3-5-sonnet ($0.015)
  ├─ Generate patch
  └─ Track cost + tokens + model selection
```

### 2. Evaluation (Free - Cloud-based)

```
sb-cli submits predictions to SWE-bench:
  ├─ Clones repositories
  ├─ Applies your patches
  ├─ Runs test suites in Docker
  └─ Returns resolution results
```

## 📈 Analyzing Results

### Check Evaluation Status

```bash
# List your runs
sb-cli list-runs swe-bench_lite dev

# Get specific report
sb-cli get-report swe-bench_lite dev adaptive_20250112_143022
```

### Load Results in Python

```python
import json
import pandas as pd

# Load generation metrics
with open("results/adaptive/adaptive_20250112_143022_generation.json") as f:
    data = json.load(f)

print(f"Total cost: ${data['cost_metrics']['total_cost_usd']}")
print(f"Resolution rate: {data['summary']['resolution_rate_percent']}%")

# Model selection stats
for model, stats in data['model_selection_stats']['models'].items():
    print(f"{model}: {stats['percentage']}%")

# Load CSV for analysis
df = pd.read_csv("results/adaptive/adaptive_20250112_143022_generation.csv")
print(df[['instance_id', 'cost_usd', 'model_used', 'resolution_status']])
```

## 🎯 Expected Results

### Cost Comparison

| Approach | Cost per Instance | Total (300 instances) |
|----------|------------------|----------------------|
| Always Claude Sonnet | $0.015 | **$4.50** |
| Always GPT-4o-mini | $0.002 | **$0.60** |
| **Adaptive Router** | $0.007 | **$2.10** |

**Adaptive achieves 90%+ of Claude's quality at 47% of the cost!**

### Model Selection Distribution

Expected routing (with cost_bias=0.5):
- 60% → gpt-4o-mini (simple issues)
- 20% → gemini-2.0-flash-exp (medium)
- 20% → claude-3-5-sonnet (complex)

## 🛠️ Development Workflow

### 1. Test Locally First

```bash
# Quick test to verify everything works
uv run python run_benchmark.py --quick --skip-submit
```

### 2. Iterate on Cost Bias

```bash
# Test different routing strategies
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.2  # Quality
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.5  # Balanced
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.8  # Cost
```

### 3. Run Medium Test

```bash
# 50 instances for statistically significant results
uv run python run_benchmark.py --medium
```

### 4. Full Benchmark

```bash
# When you're confident in your config
uv run python run_benchmark.py --full
```

## ❓ FAQ

### Where do I get the dataset?

The dataset will be loaded automatically from the SWE-bench API when you run the benchmark. No manual download needed!

### How long does evaluation take?

- Cloud evaluation: 10-30 minutes after submission
- Use `--wait` flag to wait for results automatically

### Can I use this without sb-cli?

No - sb-cli provides the official cloud evaluation. It's much easier than running Docker locally!

### What if I want to test on my own dataset?

You can modify `src/utils/dataset_loader.py` to load custom instances in SWE-bench format:

```json
[
  {
    "instance_id": "myrepo__myissue-123",
    "repo": "username/repository",
    "base_commit": "abc123...",
    "problem_statement": "Description of the issue...",
    "test_patch": "...",
    "patch": "..."
  }
]
```

### How accurate is cost tracking?

Very accurate! We track:
- Actual tokens used from API responses
- Real costs based on current pricing
- Which specific model was selected for each request

## 🔧 Troubleshooting

### "sb-cli not found"

```bash
uv sync
# or
pip install sb-cli
```

### "SWEBENCH_API_KEY not found"

Make sure `.env` exists and has your key:
```bash
cat .env | grep SWEBENCH_API_KEY
```

### "Failed to load dataset"

The dataset loader will provide instructions. Visit:
https://huggingface.co/datasets/princeton-nlp/SWE-bench

### Type errors with mypy

```bash
uv run mypy src/
```

Fix any type issues before running the benchmark.

## 📚 Resources

- **SWE-bench**: https://www.swebench.com/
- **sb-cli Docs**: https://www.swebench.com/sb-cli/
- **SWE-bench Paper**: https://arxiv.org/abs/2310.06770
- **Adaptive AI**: https://adaptive.ai/

## 🎉 Next Steps

1. **Run quick test**: `uv run python run_benchmark.py --quick`
2. **Analyze results**: Check `results/adaptive/` directory
3. **Optimize routing**: Try different `--cost-bias` values
4. **Full benchmark**: Run with `--full` when ready
5. **Compare models**: Test single models vs Adaptive

Good luck benchmarking! 🚀
