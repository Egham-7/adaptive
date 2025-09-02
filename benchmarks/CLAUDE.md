# Benchmarks - Performance Testing Suite

## Memory and Documentation

**IMPORTANT**: When working on this service, remember to:

### Memory Management
Use ByteRover MCP for persistent memory across sessions:
- **Before adding memories**: Always search first with `mcp__byterover-mcp__byterover-retrieve-knowledge` to avoid duplicates
- **Add memories**: Use `mcp__byterover-mcp__byterover-store-knowledge` for benchmark results, performance insights, troubleshooting solutions
- **Search memories**: Use `mcp__byterover-mcp__byterover-retrieve-knowledge` to recall previous conversations and solutions
- **Best practices for memory storage**: Only commit meaningful, reusable information like performance patterns, benchmarking configurations, optimization strategies, test methodologies, and results analysis that provide value beyond common knowledge

### Documentation
For documentation needs, use Ref MCP tools:
- **Search docs**: Use `mcp__Ref__ref_search_documentation` for GenAI-Perf, Python testing, MMLU benchmarking documentation
- **Read specific docs**: Use `mcp__Ref__ref_read_url` to read documentation pages

## Overview

The benchmarks service provides comprehensive performance testing and evaluation tools for the Adaptive LLM infrastructure. It includes three specialized benchmark suites: GenAI performance testing, MMLU academic evaluation, and model selection testing for validation of the AI service routing decisions.

## Key Features

- **GenAI Performance Testing**: Load testing with NVIDIA's GenAI-Perf for throughput and latency analysis
- **MMLU Academic Benchmarking**: Massive Multitask Language Understanding evaluation for model quality assessment
- **Model Selection Testing**: Validation of adaptive_ai service routing and model selection decisions
- **Concurrency Analysis**: Multi-level concurrency testing (1, 10, 25+ concurrent requests)
- **Response Quality Evaluation**: Automated quality scoring and comparison across providers
- **Performance Metrics**: Comprehensive latency, throughput, and reliability measurements

## Technology Stack

- **GenAI-Perf**: NVIDIA's performance benchmarking tool for AI inference
- **Python**: 3.10+ for scripting and analysis
- **Jupyter Notebooks**: Interactive analysis and visualization
- **Requests**: HTTP client for API testing
- **Statistics**: Built-in statistical analysis for performance metrics
- **Data Visualization**: Matplotlib and Plotly for performance charts

## Project Structure

```
benchmarks/
├── genai-perf/                      # GenAI performance benchmarking
│   ├── main.py                      # GenAI-Perf test runner
│   ├── pyproject.toml              # Dependencies and configuration
│   ├── uv.lock                     # Dependency lock file
│   ├── results/                    # Benchmark results and artifacts
│   │   ├── plots/                  # Performance visualization charts
│   │   │   ├── latency_metrics.png
│   │   │   └── throughput_vs_concurrency.png
│   │   └── *_artifacts/            # Test artifacts by scenario
│   └── README.md                   # GenAI-Perf specific documentation
│
├── MMLU benchmark/                  # Academic benchmarking
│   ├── main.py                     # MMLU evaluation runner
│   ├── api_testing.ipynb           # API testing notebook
│   ├── model_selection_analysis.ipynb # Model selection analysis notebook
│   ├── pyproject.toml              # Dependencies
│   ├── uv.lock                     # Dependency lock file
│   ├── mmlu_analysis_20250713_221207.png  # Analysis results
│   └── README.md                   # MMLU specific documentation
│
└── model-selection-testing/         # Model selection validation testing
    ├── main.py                     # Model selection testing runner
    ├── model_selection_testing.ipynb # Interactive testing notebook
    ├── pyproject.toml              # Dependencies
    ├── uv.lock                     # Dependency lock file
    ├── results/                    # Protocol testing results
    │   ├── adaptive_ai_test_report_20250716_180857.txt
    │   └── adaptive_ai_test_report_20250719_102301.txt
    ├── model_selection_testing.egg-info/  # Package metadata
    └── README.md                   # Model selection testing documentation
```

## Environment Configuration

### Required Environment Variables

```bash
# Adaptive Services
ADAPTIVE_API_BASE_URL="http://localhost:8080"    # Adaptive backend URL
ADAPTIVE_AI_BASE_URL="http://localhost:8000"     # Adaptive AI service URL

# Test Configuration
BENCHMARK_OUTPUT_DIR="./results"                 # Results output directory
BENCHMARK_DURATION=300                           # Test duration in seconds
MAX_CONCURRENCY=25                               # Maximum concurrent requests

# Provider API Keys (for direct comparison testing)
OPENAI_API_KEY="sk-..."                         # OpenAI API key
ANTHROPIC_API_KEY="sk-ant-..."                  # Anthropic API key
GOOGLE_AI_API_KEY="..."                         # Google AI API key
```

### Optional Environment Variables

```bash
# Performance Configuration
REQUEST_TIMEOUT=60                               # Request timeout in seconds
WARMUP_REQUESTS=10                              # Number of warmup requests
COOL_DOWN_TIME=5                                # Cool down time between tests

# Logging and Debugging
DEBUG_MODE=false                                # Enable debug logging
LOG_LEVEL=INFO                                  # Logging level
SAVE_RAW_RESPONSES=false                        # Save all API responses

# Visualization
FIGURE_DPI=300                                  # Chart resolution
EXPORT_FORMAT=png                               # Chart export format
```

## Benchmark Suites

### 1. GenAI-Perf Performance Testing

**Directory**: `genai-perf/`
**Purpose**: Load testing and performance analysis using NVIDIA's GenAI-Perf tool

#### Features
- **Load Testing**: Simulate high-concurrency request patterns
- **Latency Analysis**: Measure response time distributions and percentiles
- **Throughput Testing**: Determine maximum requests per second capacity
- **Concurrency Scaling**: Test performance across different concurrency levels
- **Resource Monitoring**: Track CPU, memory, and network utilization

#### Test Scenarios
- **Simple Quick Response** (C1, C10, C25): Short prompts, fast responses
- **Simple Medium Response** (C1, C10, C25): Medium complexity prompts
- **Simple Long Response** (C1, C10, C25): Complex prompts, longer responses

#### Development Commands
```bash
# Navigate to genai-perf directory
cd benchmarks/genai-perf/

# Install dependencies
uv install

# Run performance benchmarks
uv run python main.py

# Run specific concurrency test
uv run genai-perf --model adaptive-llm --concurrency 10 --num-prompts 100

# Generate performance reports
uv run python -m genai_perf.report --input results/
```

#### Key Metrics
- **Latency**: P50, P95, P99 response times
- **Throughput**: Requests per second (RPS)
- **Error Rate**: Failed request percentage
- **Resource Utilization**: CPU, memory, network usage

### 2. MMLU Academic Benchmarking

**Directory**: `MMLU benchmark/`
**Purpose**: Academic evaluation using Massive Multitask Language Understanding dataset

#### Features
- **Academic Evaluation**: Standardized testing across multiple domains
- **Quality Assessment**: Objective scoring of model responses
- **Provider Comparison**: Compare quality across different providers
- **Task Analysis**: Performance breakdown by academic subject
- **Routing Validation**: Verify AI service makes optimal model selections

#### Test Categories
- **STEM**: Mathematics, physics, chemistry, biology
- **Humanities**: History, philosophy, literature
- **Social Sciences**: Psychology, sociology, political science
- **Professional**: Law, medicine, business

#### Development Commands
```bash
# Navigate to MMLU benchmark directory
cd "benchmarks/MMLU benchmark/"

# Install dependencies
uv install

# Run MMLU evaluation
uv run python main.py

# Launch interactive analysis
uv run jupyter notebook api_testing.ipynb

# Run model selection analysis
uv run jupyter notebook model_selection_analysis.ipynb
```

#### Key Metrics
- **Overall Accuracy**: Percentage of correct answers
- **Subject Breakdown**: Performance by academic domain
- **Model Comparison**: Accuracy comparison across providers
- **Cost-Quality Ratio**: Quality per dollar spent

### 3. Model Selection Testing

**Directory**: `model-selection-testing/`
**Purpose**: Validation of adaptive_ai service routing and model selection decisions

#### Features
- **Decision Validation**: Verify AI service makes optimal routing decisions
- **Model Selection Testing**: Test task-to-model mapping accuracy
- **Cost Optimization Validation**: Verify cost savings are achieved
- **Fallback Testing**: Test provider fallback mechanisms
- **Performance Regression**: Detect performance degradations

#### Test Types
- **Classification Accuracy**: Verify prompt classification correctness
- **Provider Selection**: Validate provider routing decisions
- **Parameter Tuning**: Test parameter optimization for different tasks
- **Edge Cases**: Test handling of unusual or malformed inputs

#### Development Commands
```bash
# Navigate to model selection testing directory
cd benchmarks/model-selection-testing/

# Install dependencies
uv install

# Run model selection validation tests
uv run python main.py

# Launch interactive testing
uv run jupyter notebook model_selection_testing.ipynb

# Run comprehensive test suite
uv run pytest tests/ -v
```

#### Key Metrics
- **Routing Accuracy**: Percentage of optimal routing decisions
- **Cost Savings**: Actual vs. projected cost reductions
- **Quality Retention**: Quality maintained with cost optimization
- **Decision Speed**: Time to make routing decisions

## Performance Analysis and Visualization

### Chart Generation

**Latency Analysis**
- Response time distributions and percentiles
- Latency vs. concurrency scaling charts
- Provider latency comparison plots

**Throughput Analysis**
- Requests per second capacity curves
- Throughput vs. concurrency relationship
- Resource utilization during peak load

**Quality Analysis**
- Accuracy scores across different test categories
- Quality vs. cost trade-off analysis
- Provider quality comparison matrices

### Interactive Dashboards

**Jupyter Notebooks**
- Real-time benchmark execution and analysis
- Interactive parameter tuning and testing
- Custom analysis and visualization creation

**Results Visualization**
- Automated chart generation from benchmark results
- Exportable reports in PNG, PDF, SVG formats
- Comparative analysis across test runs

## Development Commands

### Environment Setup
```bash
# Install all benchmark dependencies
for dir in genai-perf "MMLU benchmark" model-selection-testing; do
    cd "benchmarks/$dir"
    uv install
    cd ../../
done

# Install GenAI-Perf tool
pip install genai-perf

# Verify installations
uv run python -c "import requests, pandas, matplotlib; print('Dependencies installed')"
```

### Running Benchmarks
```bash
# Run all benchmark suites
./run_all_benchmarks.sh

# Run specific benchmark suite
cd benchmarks/genai-perf/ && uv run python main.py
cd "benchmarks/MMLU benchmark/" && uv run python main.py
cd benchmarks/model-selection-testing/ && uv run python main.py

# Run with custom parameters
CONCURRENCY=50 DURATION=600 uv run python main.py
```

### Analysis and Reporting
```bash
# Generate performance reports
python scripts/generate_report.py --input benchmarks/results/

# Compare benchmark runs
python scripts/compare_results.py --baseline run1 --current run2

# Export visualizations
python scripts/export_charts.py --format pdf --output reports/
```

## Integration with CI/CD

### Automated Benchmarking
```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Benchmarks
        run: |
          docker-compose up -d
          cd benchmarks && ./run_all_benchmarks.sh
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results/
```

### Performance Regression Detection
- **Automated Alerts**: Notify on performance degradation
- **Baseline Comparison**: Compare against historical performance
- **Threshold Monitoring**: Alert when metrics exceed acceptable ranges
- **Trend Analysis**: Identify performance trends over time

## Troubleshooting

### Common Issues

**GenAI-Perf installation issues**
- Install NVIDIA drivers if using GPU testing
- Verify Python version compatibility (3.8+)
- Check system dependencies: `sudo apt-get install build-essential`

**Connection timeouts during load testing**
- Increase request timeout values
- Verify network connectivity to test endpoints
- Check if services can handle the specified concurrency
- Monitor resource utilization on target services

**MMLU evaluation failures**
- Verify dataset download and integrity
- Check API key permissions and rate limits
- Ensure sufficient disk space for results
- Validate model accessibility for evaluation

**Jupyter notebook issues**
- Install Jupyter dependencies: `uv add jupyter`
- Check port availability for notebook server
- Verify file permissions for result directories
- Clear notebook output if experiencing memory issues

### Debug Commands
```bash
# Test service connectivity
curl -f http://localhost:8080/health
curl -f http://localhost:8000/predict

# Check benchmark tool installations
genai-perf --version
python -c "import requests, pandas; print('Tools ready')"

# Run minimal benchmark test
cd benchmarks/genai-perf/
uv run genai-perf --model test --concurrency 1 --num-prompts 5

# Validate model selection testing
cd benchmarks/model-selection-testing/
uv run python -c "from main import test_classification; test_classification()"
```

## Performance Baselines and Targets

### Expected Performance Targets

**Latency Targets**
- P50 latency: <500ms for simple requests
- P95 latency: <2000ms for complex requests
- P99 latency: <5000ms for all requests

**Throughput Targets**
- Minimum: 100 RPS with 95% success rate
- Target: 1000 RPS with 99% success rate
- Peak: 2000+ RPS during load spikes

**Quality Targets**
- MMLU overall accuracy: >80% for general tasks
- Task-specific accuracy: >90% for specialized tasks
- Cost optimization: 30-70% savings vs. premium-only routing

### Benchmark Frequency
- **Continuous**: Every commit/PR for critical changes
- **Daily**: Full benchmark suite execution
- **Weekly**: Comprehensive MMLU evaluation
- **Monthly**: Extended duration stress testing

## Contributing

### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Notebooks**: Clear markdown documentation and clean outputs
- **Scripts**: Comprehensive error handling and logging
- **Documentation**: Update README files for new benchmark types

### Testing Requirements
- **Unit Tests**: Test utility functions and data processing
- **Integration Tests**: Validate end-to-end benchmark execution
- **Performance Tests**: Regression testing for benchmark tools
- **Data Validation**: Verify result accuracy and completeness

### Documentation Updates
**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new benchmark suites or test scenarios
   - Modifying performance targets or expected baselines
   - Changing test configurations or environment variables
   - Adding new metrics or analysis capabilities
   - Updating dependencies or tool versions

2. **Update individual benchmark README files** when:
   - Adding new test cases or scenarios
   - Modifying benchmark-specific configurations
   - Changing analysis procedures or methodologies

3. **Update root CLAUDE.md** when:
   - Adding new benchmark capabilities that affect the overall architecture
   - Changing the role of benchmarking in the development workflow

### Pull Request Process
1. Create feature branch from `dev`
2. Implement changes with appropriate tests
3. Run benchmark validation: `./validate_benchmarks.sh`
4. **Update relevant documentation** (CLAUDE.md files, README files)
5. Submit PR with benchmark results and performance impact analysis
6. Include baseline comparisons showing performance changes