# Go LLM API Benchmarking with GenAI-Perf

This directory contains a complete benchmarking solution for your Go LLM API using NVIDIA's GenAI-Perf tool. The setup is designed to be replicable and provides comprehensive performance analysis.

## 📁 Directory Structure

```
benchmarks/genai-perf/
├── docker-compose.yml          # GenAI-Perf container configuration
├── run_benchmarks.sh           # Complete workflow script
├── scripts/
│   ├── benchmark_go_api.sh     # Main benchmarking script
│   └── analyze_results.py      # Results analysis script
├── results/                    # Generated benchmark results
│   ├── plots/                  # Performance visualization plots
│   ├── benchmark_summary_report.txt
│   └── go_api_benchmark_results.csv
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Docker**: Ensure Docker is installed and running
2. **Go API**: Your Go LLM API should be running and accessible
3. **NVIDIA GPU**: Required for GenAI-Perf (with NVIDIA drivers)

### Basic Usage

```bash
# Navigate to benchmarks directory
cd benchmarks/genai-perf

# Run benchmarks with default settings
./run_benchmarks.sh

# Run with custom API URL
./run_benchmarks.sh -u localhost:8080

# Run all benchmark modes
./run_benchmarks.sh -b 4
```

### Configuration Options

```bash
./run_benchmarks.sh [OPTIONS]

Options:
  -u, --url URL          Router URL (default: localhost:8080)
  -m, --model MODEL      Model name (default: adaptive-go-api)
  -b, --benchmark MODE   Benchmark mode (default: 2)
  -h, --help             Show help message

Benchmark Modes:
  1: Full benchmark with synthetic tokens (requires tokenizer)
  2: Simple benchmark (no tokenizer required) - RECOMMENDED
  3: Load test only
  4: All tests
```

## 📊 Benchmark Modes

### Mode 1: Full Benchmark
- Uses synthetic token generation
- Tests various input/output token combinations
- Requires tokenizer support
- Best for comprehensive analysis

### Mode 2: Simple Benchmark (Default)
- No tokenizer dependencies
- Tests different response lengths
- Faster execution
- Recommended for most use cases

### Mode 3: Load Test Only
- Sustained load testing
- Single comprehensive test
- Good for capacity planning

### Mode 4: All Tests
- Runs all above modes
- Most comprehensive analysis
- Longest execution time

## 🔧 Manual Execution

If you prefer to run components manually:

### 1. Start Container
```bash
docker-compose up -d
```

### 2. Run Benchmarks
```bash
# Install dependencies
docker-compose exec genai-perf pip install matplotlib seaborn pandas numpy

# Run benchmark script
docker-compose exec genai-perf bash scripts/benchmark_go_api.sh 2

# Analyze results
docker-compose exec genai-perf python scripts/analyze_results.py
```

### 3. Cleanup
```bash
docker-compose down
```

## 📈 Understanding Results

### Key Metrics

- **Throughput (TPS)**: Tokens per second your API can process
- **TTFT**: Time to First Token - critical for user experience
- **ITL**: Inter-Token Latency - affects streaming quality
- **E2E Latency**: End-to-end request latency

### Generated Outputs

1. **Performance Plots**: Visual charts showing performance curves
2. **Summary Report**: Text-based analysis with recommendations
3. **CSV Data**: Raw benchmark data for further analysis

### Result Files

- `benchmark_summary_report.txt`: Comprehensive performance analysis
- `go_api_benchmark_results.csv`: Raw benchmark data
- `plots/`: Directory containing performance visualizations
  - `throughput_vs_concurrency.png`
  - `latency_metrics.png`
  - `latency_vs_throughput.png`
  - `performance_heatmap.png`

## 🛠️ Customization

### API Endpoint Configuration

Edit the environment variables in `docker-compose.yml`:

```yaml
environment:
  - ROUTER_URL=localhost:8080    # Your API URL
  - MODEL_NAME=adaptive-go-api   # Model identifier
```

### Test Scenarios

Modify test cases in `scripts/benchmark_go_api.sh`:

```bash
declare -A testCases
testCases["Quick_Chat"]="50/25"
testCases["Code_Generation"]="100/200"
testCases["Custom_Test"]="150/300"
```

### Concurrency Levels

Adjust concurrency testing in the benchmark script:

```bash
for concurrency in 1 2 5 10 20 50; do
    # Add or remove concurrency levels as needed
done
```

## 🔍 Troubleshooting

### Common Issues

1. **API Not Accessible**
   ```bash
   # Check if your Go API is running
   curl -s http://localhost:8080/health
   
   # Update ROUTER_URL if needed
   export ROUTER_URL=localhost:3000
   ```

2. **Docker Issues**
   ```bash
   # Ensure Docker is running
   docker info
   
   # Check container logs
   docker-compose logs genai-perf
   ```

3. **NVIDIA GPU Issues**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Verify Docker can access GPU
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

### Performance Issues

- **Low Throughput**: Check API scaling, database connections, or model loading
- **High Latency**: Examine network latency, API processing time, or resource constraints
- **Memory Issues**: Monitor container memory usage and adjust limits

## 📝 Best Practices

### Before Benchmarking

1. **Warm up your API**: Ensure it's fully loaded and ready
2. **Stable environment**: Close unnecessary applications
3. **Consistent hardware**: Use the same machine for comparative tests
4. **Network stability**: Ensure stable network connection

### During Benchmarking

1. **Monitor resources**: Watch CPU, memory, and GPU usage
2. **Consistent load**: Avoid other heavy operations
3. **Multiple runs**: Run benchmarks multiple times for accuracy

### After Benchmarking

1. **Compare results**: Look for patterns across different runs
2. **Identify bottlenecks**: Focus on metrics that matter most
3. **Document findings**: Save results with context and environment details

## 🔄 Reproducibility

To ensure reproducible results:

1. **Version control**: Document API version, GenAI-Perf version
2. **Environment documentation**: Record hardware specs, OS version
3. **Seed values**: Use consistent random seeds where possible
4. **Test isolation**: Run tests in isolated environments

## 📚 Additional Resources

- [GenAI-Perf Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/perf_toolkit.html)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [Performance Optimization Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html)

## 🤝 Contributing

To improve this benchmarking setup:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This benchmarking setup is part of the adaptive project. Please refer to the main project license.

---

**Note**: This benchmarking setup is designed for development and testing purposes. For production benchmarking, consider additional factors like security, compliance, and enterprise-grade monitoring.