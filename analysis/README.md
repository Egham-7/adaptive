# Adaptive Analysis

Analysis tools and pipelines for cost analysis, performance optimization, and usage metrics in the Adaptive AI platform.

## Overview

This package contains analysis tools for understanding the financial and performance impact of different AI model usage patterns.

### Features

- **Cost Analysis**: Financial impact analysis for different AI model usage patterns
- **Performance Analysis**: Optimization tools and performance metrics
- **Usage Metrics**: Reporting utilities and resource utilization analysis
- **Resource Monitoring**: System and service resource tracking

## Quick Start

```bash
# Install with Poetry
cd analysis
poetry install

# Run cost analysis
poetry run python -m cost_analysis.cost_analysis_10k.cost_analysis_pipeline
```

## Usage

### Cost Analysis

Analyze the financial impact of different AI model usage patterns:

```python
from cost_analysis.cost_analysis_10k.cost_analysis_pipeline import run_cost_analysis

# Run cost analysis for 10k requests
results = run_cost_analysis(num_requests=10000)
```

### Running Analysis Scripts

```bash
# Activate virtual environment
poetry shell

# Run cost analysis pipeline
python -m cost_analysis.cost_analysis_10k.cost_analysis_pipeline
```

## Development

```bash
# Install dependencies including dev tools
poetry install --with dev

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov

# Code quality checks
poetry run black .          # Format code
poetry run ruff check .     # Lint code
poetry run mypy .           # Type checking
```

## Project Structure

```
analysis/
├── cost_analysis/          # Cost analysis tools
│   └── cost_analysis_10k/  # 10k request cost analysis
├── tests/                  # Test files
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Available Analysis Tools

### Cost Analysis 10k

Located in `cost_analysis/cost_analysis_10k/`, this tool:
- Analyzes cost patterns for 10,000 request scenarios
- Generates cost comparison charts
- Provides optimization recommendations
- Outputs visual reports (PNG format)

## Contributing

1. Follow existing code style and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for new features
4. Run quality checks before submitting changes

## Requirements

- Python 3.8+
- Poetry for dependency management
- See `pyproject.toml` for complete dependencies