# HumanEval Benchmarks with DeepEval

Professional-grade HumanEval benchmarking suite for evaluating code generation models with comprehensive **per-row cost and token tracking**.

## Features

- **Three Model Implementations**: Claude 4.5 Sonnet, GLM-4.6, and Adaptive routing
- **Response-Based Cost Tracking**: Extracts actual costs and token counts from API responses
- **Per-Row Granularity**: Tracks cost, tokens, and latency for every single sample (all 32,800 samples in full benchmark)
- **Comprehensive Reporting**: JSON, CSV, and Markdown reports with detailed analytics
- **Model Comparison**: Cross-model performance, cost, and efficiency analysis
- **Adaptive Routing**: Track which models are selected for each request
- **DeepEval Integration**: Full HumanEval benchmark support with pass@k metrics

## Project Structure

```
benchmarks/
├── humaneval/
│   ├── models/                    # Model implementations
│   │   ├── base.py               # Base classes and data structures
│   │   ├── claude_model.py       # Claude 4.5 Sonnet
│   │   ├── glm_model.py          # GLM-4.6
│   │   └── adaptive_model.py     # Adaptive routing
│   ├── tests/                    # Pytest test suites
│   │   ├── test_claude.py        # Claude benchmarks
│   │   ├── test_glm.py           # GLM benchmarks
│   │   ├── test_adaptive.py      # Adaptive benchmarks
│   │   └── test_comparison.py    # Cross-model comparison
│   ├── utils/                    # Utilities
│   │   ├── response_parser.py    # API response parsing
│   │   ├── result_tracker.py     # Results storage
│   │   └── reporting.py          # Report generation
│   ├── config.py                 # Configuration management
│   └── run_benchmark.py          # CLI entry point
├── results/                      # Benchmark results
│   ├── claude/                   # Claude results
│   ├── glm/                      # GLM results
│   ├── adaptive/                 # Adaptive results
│   └── comparisons/              # Comparison reports
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
# Navigate to benchmarks directory
cd benchmarks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env.local` and add your API keys:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```bash
# API Keys
ANTHROPIC_API_KEY=your-anthropic-key-here
GLM_API_KEY=your-glm-key-here
ADAPTIVE_API_KEY=your-adaptive-key-here

# Optional: Customize benchmark settings
HUMANEVAL_N_SAMPLES=200
HUMANEVAL_K_VALUE=10
```

### 3. Run Benchmarks

#### Using CLI (Recommended)

```bash
# Quick test (5 tasks, 50 samples) - ~5-10 minutes
python humaneval/run_benchmark.py claude --quick
python humaneval/run_benchmark.py glm --quick
python humaneval/run_benchmark.py adaptive --quick

# Full benchmark (164 tasks, 200 samples) - ~2-4 hours
python humaneval/run_benchmark.py claude --full
python humaneval/run_benchmark.py glm --full
python humaneval/run_benchmark.py adaptive --full

# Run all models
python humaneval/run_benchmark.py all --quick

# Compare existing results
python humaneval/run_benchmark.py compare
```

#### Using Pytest

```bash
# Run specific model test
pytest humaneval/tests/test_claude.py::TestClaudeHumanEval::test_claude_humaneval_quick -v -s

# Run full benchmark
pytest humaneval/tests/test_claude.py::TestClaudeHumanEval::test_claude_humaneval_full -v -s

# Run comparison
pytest humaneval/tests/test_comparison.py -v -s
```

#### Using DeepEval CLI

```bash
# Run with deepeval
deepeval test run humaneval/tests/test_claude.py -v
```

## Cost Tracking

### How It Works

All cost and token information is **extracted directly from API responses**, ensuring accuracy:

1. **API Response Parsing**: Each model wrapper extracts usage data from the API response
2. **Per-Sample Tracking**: Every single code generation sample is tracked individually
3. **Task Aggregation**: Costs are summed across all samples for each task
4. **Overall Statistics**: Total costs and tokens are calculated across all 164 tasks

### Example Output

```json
{
  "model": "claude-sonnet-4-5",
  "overall_score": 0.847,
  "pass_at_k": {"k": 10, "score": 0.847},
  "total_cost_usd": 45.67,
  "total_tokens": 5234567,
  "per_task_results": [
    {
      "task_id": "HumanEval/0",
      "total_cost_usd": 0.278,
      "total_input_tokens": 7520,
      "total_output_tokens": 15400,
      "samples_generated": 200,
      "per_sample_costs": [0.00139, 0.00141, 0.00138, ...]
    }
  ]
}
```

### Cost Estimation

**Quick Test** (5 tasks, 50 samples):
- Claude Sonnet 4.5: ~$2-3
- GLM-4: Varies by pricing
- Adaptive: Depends on selected models

**Full Benchmark** (164 tasks, 200 samples):
- Claude Sonnet 4.5: ~$40-60
- GLM-4: Varies by pricing
- Adaptive: Typically 20-40% cheaper than single model

## Model Implementations

### Claude 4.5 Sonnet

```python
from humaneval.models import ClaudeForDeepEval

model = ClaudeForDeepEval(
    model_name="claude-sonnet-4-5",
    api_key="your-key"
)
```

**Pricing**: $3 per 1M input tokens, $15 per 1M output tokens

### GLM-4.6

```python
from humaneval.models import GLMForDeepEval

model = GLMForDeepEval(
    model_name="glm-4",
    api_key="your-key",
    api_base="https://open.bigmodel.cn/api/paas/v4/"
)
```

### Adaptive Routing

```python
from humaneval.models import AdaptiveForDeepEval

model = AdaptiveForDeepEval(
    api_key="your-key",
    models=[
        'openai:gpt-5-mini',
        'anthropic:claude-sonnet-4-5',
        'gemini:gemini-2.5-flash-lite'
    ],
    cost_bias=0.5  # 0=cheapest, 1=best performance
)
```

The Adaptive model automatically selects the best model for each request and tracks which models were chosen.

## Results and Reports

### Output Files

Each benchmark run generates:

1. **Detailed JSON** (`model_timestamp.json`): Complete results with per-sample costs
2. **Summary JSON** (`model_timestamp_summary.json`): Aggregated results without per-sample arrays
3. **CSV** (`model_timestamp.csv`): Task-level summary for spreadsheet analysis
4. **Markdown Report** (`model_timestamp_report.md`): Human-readable report

### Results Location

```
results/
├── claude/
│   ├── claude-sonnet-4-5_20251110_143022.json
│   ├── claude-sonnet-4-5_20251110_143022_summary.json
│   ├── claude-sonnet-4-5_20251110_143022.csv
│   └── claude-sonnet-4-5_20251110_143022_report.md
├── glm/
│   └── ...
├── adaptive/
│   └── ...
└── comparisons/
    └── comparison_20251110_150000.md
```

### Viewing Results

```bash
# View JSON results
cat results/claude/claude-sonnet-4-5_*.json | jq '.total_cost_usd'

# View CSV in spreadsheet
open results/claude/claude-sonnet-4-5_*.csv

# Read markdown report
cat results/claude/claude-sonnet-4-5_*_report.md
```

## Comparison Reports

Generate cross-model comparisons:

```bash
python humaneval/run_benchmark.py compare
```

The comparison report includes:
- Performance ranking (pass@k scores)
- Cost analysis
- Cost efficiency (score per dollar)
- Token usage comparison
- Model selection stats (for Adaptive)

## Configuration

### Environment Variables

See `.env.example` for all available options:

- `HUMANEVAL_N_SAMPLES`: Samples per task (default: 200)
- `HUMANEVAL_K_VALUE`: Pass@k metric (default: 10)
- `BENCHMARK_PARALLEL`: Enable parallel execution (default: true)
- `MAX_CONCURRENT_TASKS`: Max parallel tasks (default: 20)

### Custom Configuration

```python
from humaneval.config import BenchmarkSettings

settings = BenchmarkSettings()
settings.humaneval.n_samples = 100  # Override
settings.humaneval.k_value = 5
settings.print_summary()
```

## Advanced Usage

### Custom Model Implementation

Extend `BaseHumanEvalModel` to add your own model:

```python
from humaneval.models.base import BaseHumanEvalModel, ResponseMetrics

class MyCustomModel(BaseHumanEvalModel):
    def generate_with_metrics(self, prompt, temperature, max_tokens):
        # Your implementation
        response = your_api_call(prompt)

        # Extract metrics from response
        metrics = ResponseMetrics(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.cost,  # From API
            model_used="my-model"
        )

        return response.text, metrics
```

### Programmatic Usage

```python
from deepeval.benchmarks import HumanEval
from humaneval.models import ClaudeForDeepEval
from humaneval.utils import ResultTracker, generate_markdown_report

# Initialize model
model = ClaudeForDeepEval()

# Run benchmark
benchmark = HumanEval(n=200)
benchmark.evaluate(model=model, k=10)

# Track results
tracker = ResultTracker("my-model")
for task_metrics in model.get_all_task_metrics():
    tracker.add_task_result(task_metrics)

tracker.set_benchmark_score(benchmark.overall_score, k=10)
benchmark_run = tracker.finalize()

# Save results
tracker.save_json(benchmark_run)
generate_markdown_report(benchmark_run)
```

## Troubleshooting

### API Key Issues

```bash
# Verify API keys are loaded
python -c "from humaneval.config import BenchmarkSettings; BenchmarkSettings().validate_model('claude')"
```

### Rate Limiting

If you hit rate limits:
- Reduce `MAX_CONCURRENT_TASKS` in `.env.local`
- Use `--quick` mode for testing
- Add retry configuration in `.env.local`

### Memory Issues

For large benchmark runs:
- The detailed JSON files can be ~50-100MB
- Use summary JSON for smaller files
- CSV exports are more memory-efficient

## Best Practices

1. **Start with Quick Tests**: Always run `--quick` mode first to validate setup
2. **Monitor Costs**: Check costs after quick tests before running full benchmarks
3. **Use Caching**: Enable `USE_CACHE=true` during development
4. **Save Results**: All results are automatically saved with timestamps
5. **Compare Models**: Run all models with same configuration for fair comparison

## HumanEval Benchmark Details

- **Total Tasks**: 164 hand-crafted Python programming challenges
- **Default Samples**: 200 per task (n=200)
- **Default Metric**: pass@10 (probability that ≥1 of top 10 samples passes tests)
- **Total API Calls**: 164 tasks × 200 samples = 32,800 calls per full benchmark
- **Evaluation**: Functional correctness (not text similarity)

## Contributing

To add a new model:

1. Create model class in `humaneval/models/`
2. Implement `generate_with_metrics()` method
3. Add test file in `humaneval/tests/`
4. Update configuration in `config.py`
5. Add CLI support in `run_benchmark.py`

## License

This benchmark suite is built on top of:
- [DeepEval](https://github.com/confident-ai/deepeval) - Evaluation framework
- [HumanEval](https://github.com/openai/human-eval) - Original dataset by OpenAI

## Support

For issues or questions:
- Check `.env.example` for configuration options
- Review logs in console output
- Examine saved JSON results for detailed error information

## Acknowledgments

- OpenAI for the HumanEval dataset
- Confident AI for the DeepEval framework
- Anthropic, GLM, and other model providers
