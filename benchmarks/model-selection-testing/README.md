# Model Selection Testing

This module tests the MinionS protocol and model selection using the routellm/gpt4_dataset from HuggingFace.

## Setup

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Development Tools

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Run all checks:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/ tests/
```

## Main Testing

The main testing is done in `model_selection_testing.ipynb` which:
1. Loads the routellm/gpt4_dataset
2. Tests model selection algorithms
3. Evaluates MinionS protocol performance
4. Generates analysis reports

## Dataset

Using routellm/gpt4_dataset from HuggingFace:
- GPT-4 generated conversations
- Various task types and complexities
- High-quality benchmark data