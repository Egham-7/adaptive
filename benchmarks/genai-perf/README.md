# GenAI Performance Benchmarking Tool

A comprehensive, type-safe Python CLI tool for benchmarking LLM APIs using NVIDIA's GenAI-Perf. This tool provides a clean interface to all GenAI-Perf parameters with automatic analysis and visualization.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.13+** with `uv` package manager
2. **GenAI-Perf**: Install via `pip install genai-perf>=0.0.15`
3. **LLM API**: Your API should be running and accessible

### Installation

```bash
# Navigate to the benchmarks directory
cd benchmarks/genai-perf

# Install dependencies
uv sync

# Run basic benchmark
uv run main.py benchmark --url "https://your-api-endpoint.com"
```

## 🛠️ Commands

### `benchmark` - Run Performance Tests

```bash
uv run main.py benchmark [OPTIONS]
```

**Core Options:**
- `--url, -u`: API endpoint URL (default: `http://localhost:8080`)
- `--model, -m`: Model name (default: `adaptive-go-api`)
- `--concurrency, -c`: Comma-separated concurrency levels (default: `1,5,10,25`)
- `--check-health/--no-check-health`: Enable/disable health check (default: enabled)

**Advanced Options:**

*Audio Input:*
- `--audio-length-mean`: Mean audio length in seconds
- `--audio-length-stddev`: Audio length standard deviation
- `--audio-format`: Audio format (`wav`, `mp3`)
- `--audio-depths`: Comma-separated audio bit depths
- `--audio-sample-rates`: Comma-separated sample rates
- `--audio-num-channels`: Number of audio channels (1 or 2)

*Image Input:*
- `--image-width-mean/--image-width-stddev`: Image width parameters
- `--image-height-mean/--image-height-stddev`: Image height parameters
- `--image-format`: Image format (`png`, `jpeg`)

*Input Configuration:*
- `--num-dataset-entries, --num-prompts`: Number of unique prompts
- `--batch-size-text/--batch-size-audio/--batch-size-image`: Batch sizes
- `--extra-inputs`: Extra inputs in `key:value` format
- `--input-file`: Custom input file path
- `--synthetic-input-tokens-mean/--synthetic-input-tokens-stddev`: Synthetic token parameters

*Output Configuration:*
- `--output-tokens-mean`: Mean number of output tokens
- `--output-tokens-stddev`: Output tokens standard deviation
- `--output-tokens-mean-deterministic`: Enable deterministic output tokens

*Profiling:*
- `--measurement-interval`: Measurement interval in milliseconds
- `--request-count`: Total number of requests
- `--request-rate`: Request rate limit
- `--stability-percentage`: Stability threshold percentage

*Session Management:*
- `--num-sessions`: Number of concurrent sessions
- `--session-concurrency`: Session-level concurrency
- `--session-turns-mean/--session-turns-stddev`: Session turn parameters
- `--session-turn-delay-mean/--session-turn-delay-stddev`: Turn delay parameters

*Advanced Features:*
- `--streaming`: Enable streaming API
- `--verbose`: Enable verbose output
- `--generate-plots`: Generate performance plots
- `--enable-checkpointing`: Enable state checkpointing
- `--tokenizer`: Custom tokenizer name or path
- `--backend`: Backend type (`tensorrtllm`, `vllm`)

### `analyze` - Analyze Results

```bash
uv run main.py analyze [OPTIONS]
```

- `--results-dir, -r`: Results directory (default: `./results`)
- `--plots-only`: Generate only plots
- `--summary-only`: Generate only summary report

### `status` - Check Results Status

```bash
uv run main.py status [--results-dir DIRECTORY]
```

### `run-all` - Complete Pipeline

```bash
uv run main.py run-all [ALL_BENCHMARK_OPTIONS]
```

Runs benchmarks and analysis in sequence.

## 📊 Example Usage

### Basic Benchmark
```bash
uv run main.py benchmark \
  --url "https://api.example.com" \
  --concurrency "1,5,10" \
  --num-prompts 50
```

### Advanced Benchmark with Custom Parameters
```bash
uv run main.py benchmark \
  --url "https://api.example.com" \
  --concurrency "1,5,10,25,50" \
  --num-prompts 100 \
  --output-tokens-mean 200 \
  --measurement-interval 10000 \
  --streaming \
  --verbose
```

### Complete Pipeline
```bash
uv run main.py run-all \
  --url "https://api.example.com" \
  --concurrency "1,10,25" \
  --num-prompts 30
```

## 📈 Output and Results

### Generated Files

- **JSON Results**: `simple_{test_type}_c{concurrency}_genai_perf.json`
- **CSV Results**: `simple_{test_type}_c{concurrency}_genai_perf.csv`
- **Plots**: `results/plots/` directory
  - `throughput_vs_concurrency.png`
  - `latency_metrics.png`
- **Summary**: `benchmark_summary_report.txt`
- **Raw Data**: `go_api_benchmark_results.csv`

### Key Metrics

- **Throughput**: Tokens per second (TPS)
- **TTFT**: Time to First Token (ms)
- **ITL**: Inter-Token Latency (ms)  
- **E2E Latency**: End-to-end request latency (ms)

## 🏗️ Architecture

### Type-Safe Design

The tool uses a `BenchmarkParameters` dataclass for type safety:

```python
@dataclass
class BenchmarkParameters:
    url: str
    model: str
    concurrency: Optional[str] = None
    # ... all other parameters with proper types
```

### Modular Structure

- **Parameter Management**: `create_benchmark_params()` for clean parameter collection
- **Shared Logic**: `_run_benchmark_with_params()` eliminates code duplication
- **Type Safety**: Full mypy compatibility with proper type hints
- **Error Handling**: Comprehensive error checking and user feedback

## 🔧 Development

### Code Quality

```bash
# Type checking
uv run mypy .

# Code formatting
uv run black .

# Linting  
uv run ruff .
```

### Test Scenarios

The tool runs three default test scenarios:
- **Quick Response**: 50 max tokens
- **Medium Response**: 150 max tokens  
- **Long Response**: 300 max tokens

Each scenario tests across all specified concurrency levels.

## 🛠️ Troubleshooting

### Common Issues

1. **API Health Check Fails**
   ```bash
   # Verify API is accessible
   curl -s https://your-api-endpoint.com/health
   ```

2. **Invalid Concurrency Format**
   ```bash
   # Use comma-separated values
   --concurrency "1,5,10,25"
   ```

3. **GenAI-Perf Not Found**
   ```bash
   # Install GenAI-Perf
   pip install genai-perf>=0.0.15
   ```

### Performance Tips

- Use `--measurement-interval` ≥ 8000ms for stable results
- Start with low concurrency levels and increase gradually
- Monitor API resource usage during benchmarks
- Use `--num-prompts` 30+ for statistical significance

## 📚 Advanced Features

### Custom Input Files

```bash
uv run main.py benchmark \
  --input-file "custom_prompts.jsonl" \
  --url "https://api.example.com"
```

### Streaming API Testing

```bash
uv run main.py benchmark \
  --streaming \
  --url "https://api.example.com"
```

### Session-Based Testing

```bash
uv run main.py benchmark \
  --num-sessions 10 \
  --session-turns-mean 3 \
  --url "https://api.example.com"
```

## 🤝 Contributing

1. Follow the existing code style and type hints
2. Add tests for new features
3. Update documentation for new parameters
4. Ensure mypy passes: `uv run mypy .`

## 📄 License

Part of the adaptive project. See main project license.