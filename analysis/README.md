# Adaptive Analysis

Analysis tools and pipelines for the Adaptive project.

## Overview

This package contains various analysis tools and pipelines including:
- Cost analysis for different AI models and usage patterns
- Performance analysis and optimization tools
- Usage metrics and reporting utilities
- Resource utilization analysis

## Installation

### Using Poetry (Recommended)

```bash
cd adaptive/analysis
poetry install
```

### Development Installation

```bash
cd adaptive/analysis
poetry install --with dev
```

## Usage

### Cost Analysis

The cost analysis pipeline helps you understand the financial impact of different AI model usage patterns:

```python
from cost_analysis.cost_analysis_10k.cost_analysis_pipeline import run_cost_analysis

# Run cost analysis for 10k requests
results = run_cost_analysis(num_requests=10000)
```

### Running Analysis Scripts

```bash
# Activate the virtual environment
poetry shell

# Run cost analysis
python -m cost_analysis.cost_analysis_10k.cost_analysis_pipeline
```

## Development

### Setup

```bash
# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

### Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov
```

### Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy .
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

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Run quality checks before submitting changes

## License

This project is part of the Adaptive ecosystem and follows the same licensing terms.