# Adaptive Benchmarks

Benchmark tests and evaluations for the Adaptive project.

## Overview

This package contains various benchmark tests and performance evaluations including:
- **ARC (Abstraction and Reasoning Corpus)** tests - Easy and Hard variants
- **Full Stack Benchmarks** - End-to-end system performance tests
- **Test Complexity Analysis** - Computational complexity evaluation tools
- **Performance Evaluation Tools** - Model and system performance metrics

## Installation

### Using Poetry (Recommended)

```bash
cd adaptive/benchmarks
poetry install
```

### Development Installation

```bash
cd adaptive/benchmarks
poetry install --with dev
```

## Usage

### Running Benchmarks

#### ARC Benchmarks

```bash
# Run ARC Easy benchmarks
poetry run run-arc-easy

# Run ARC Hard benchmarks  
poetry run run-arc-hard
```

#### Full Stack Benchmarks

```bash
# Run full stack performance tests
poetry run run-fullstack-bench
```

#### Complexity Tests

```bash
# Run computational complexity analysis
poetry run run-complexity-test
```

### Using as a Library

```python
from arc_easy import run_arc_easy_benchmark
from arc_hard import run_arc_hard_benchmark
from fullstackbench import run_fullstack_benchmark

# Run specific benchmarks
easy_results = run_arc_easy_benchmark()
hard_results = run_arc_hard_benchmark()
fullstack_results = run_fullstack_benchmark()
```

### Benchmark Configuration

Most benchmarks can be configured through environment variables or configuration files:

```bash
# Set number of test samples
export BENCHMARK_SAMPLES=1000

# Set timeout for individual tests (seconds)
export BENCHMARK_TIMEOUT=300

# Set output directory for results
export BENCHMARK_OUTPUT_DIR="./results"
```

## Development

### Setup

```bash
# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests (excluding benchmarks)
poetry run pytest

# Run tests with coverage
poetry run pytest --cov

# Run benchmark tests (can be slow)
poetry run pytest --benchmark-enable

# Run only fast tests
poetry run pytest -m "not slow"

# Run specific benchmark categories
poetry run pytest -m "arc"
poetry run pytest -m "fullstack"
```

### Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy .

# Run all quality checks
poetry run black . && poetry run ruff check . && poetry run mypy .
```

### Adding New Benchmarks

1. Create a new module in the appropriate directory
2. Implement the benchmark following the existing patterns
3. Add tests in the `tests/` directory
4. Update the `pyproject.toml` scripts section if needed
5. Document the benchmark in this README

Example benchmark structure:

```python
from typing import Dict, Any
import time
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    duration: float
    success: bool
    metrics: Dict[str, Any]

def run_my_benchmark() -> BenchmarkResult:
    """Run a custom benchmark."""
    start_time = time.time()
    
    # Your benchmark logic here
    success = True
    metrics = {"accuracy": 0.95, "throughput": 100}
    
    duration = time.time() - start_time
    
    return BenchmarkResult(
        name="my_benchmark",
        duration=duration,
        success=success,
        metrics=metrics
    )
```

## Project Structure

```
benchmarks/
├── arc_easy/              # ARC Easy benchmark tests
├── arc_hard/              # ARC Hard benchmark tests  
├── fullstackbench/        # Full stack performance tests
├── test_complexity/       # Complexity analysis tools
├── tests/                 # Test files
├── results/               # Benchmark results (generated)
├── pyproject.toml        # Project configuration
├── README.md             # This file
├── final_prompt_code_generation.py  # Code generation benchmarks
├── vector_analysis_report_easy.md   # Easy vector analysis results
└── vector_analysis_report_medium.md # Medium vector analysis results
```

## Benchmark Categories

### ARC (Abstraction and Reasoning Corpus)

Tests the system's ability to:
- Learn abstract patterns from few examples
- Apply learned patterns to new situations  
- Handle visual reasoning tasks
- Generalize across different problem types

### Full Stack Benchmarks

Evaluates end-to-end system performance:
- API response times
- Database query performance
- Memory usage patterns
- Concurrent user handling
- Error rates and recovery

### Complexity Analysis

Measures computational characteristics:
- Time complexity of algorithms
- Space complexity patterns
- Scaling behavior with input size
- Resource utilization efficiency

## Results and Reporting

Benchmark results are automatically saved to the `results/` directory with timestamps. Results include:

- Performance metrics (latency, throughput, accuracy)
- System resource usage
- Error rates and failure modes
- Comparative analysis against baselines
- Visualizations and plots

### Viewing Results

Results are saved in multiple formats:
- JSON for programmatic access
- CSV for data analysis
- HTML reports for human review
- PNG/SVG plots for visualization

## Continuous Integration

Benchmarks are integrated into the CI/CD pipeline:
- Light benchmarks run on every commit
- Full benchmark suites run nightly
- Performance regression detection
- Automated performance reports

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new benchmarks
3. Update documentation and examples
4. Ensure benchmarks are deterministic when possible
5. Include performance baselines and expectations

## Performance Baselines

Current performance targets:
- ARC Easy: >80% accuracy in <5 minutes
- ARC Hard: >60% accuracy in <15 minutes  
- Full Stack: <100ms p95 latency at 1000 RPS
- Memory: <2GB peak usage for standard tests

## License

This project is part of the Adaptive ecosystem and follows the same licensing terms.