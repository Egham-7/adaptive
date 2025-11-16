# Adaptive Benchmarks

This directory contains benchmarking suites for evaluating the Adaptive router against various industry-standard benchmarks.

## Available Benchmarks

### SWE-bench

Professional-grade SWE-bench Lite benchmarking suite for evaluating software engineering capabilities with comprehensive **per-instance cost and token tracking**.

See [swe-bench/README.md](swe-bench/README.md) for detailed documentation.

#### Quick Start

```bash
cd swe-bench
./quick_start.sh
```

## Project Structure

```
benchmarks/
├── swe-bench/              # SWE-bench Lite benchmark suite
│   ├── src/                # Source code
│   ├── data/               # Dataset storage
│   ├── predictions/        # Model predictions
│   ├── results/            # Benchmark results
│   └── README.md           # Full documentation
├── README.md               # This file
└── __init__.py             # Package marker
```

## Requirements

Each benchmark has its own dependencies managed through `pyproject.toml` and `uv.lock` files in its respective directory.

## Adding New Benchmarks

To add a new benchmark:

1. Create a new directory under `benchmarks/`
2. Add a comprehensive `README.md` with setup and usage instructions
3. Include `pyproject.toml` for dependency management
4. Ensure all code passes `mypy`, `ruff`, and `black` checks
5. Update this README with a link to the new benchmark

## Development

All benchmarks should maintain high code quality standards:

- **Type checking**: `mypy` for static type analysis
- **Linting**: `ruff` for code quality checks
- **Formatting**: `black` for consistent code style

## Support

For issues or questions about specific benchmarks, refer to the README in each benchmark's directory.
