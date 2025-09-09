# Test Suite Documentation

This document describes the test suite organization for the adaptive_ai service, including test categories, structure, and execution methods.

## Test Structure

The test suite is organized by architectural layers and testing concerns:

```
tests/
├── README.md                      # This documentation
├── conftest.py                    # Global pytest configuration
├── fixtures/                      # Reusable test fixtures
│   ├── __init__.py
│   ├── config_fixtures.py         # Configuration and settings fixtures
│   ├── model_fixtures.py          # Model and schema fixtures
│   ├── request_fixtures.py        # API request/response fixtures
│   └── ml_fixtures.py             # ML model and service mocks
├── unit/                          # Unit tests (fast, isolated, no external deps)
│   ├── __init__.py
│   ├── core/                      # Core functionality tests
│   │   ├── __init__.py
│   │   └── test_config.py         # Configuration loading and validation
│   ├── models/                    # Data model tests
│   │   ├── __init__.py
│   │   ├── test_core_models.py    # Core request/response models
│   │   ├── test_classification_models.py  # Classification result models
│   │   └── test_enums.py          # Enum validation and behavior
│   ├── services/                  # Service layer tests
│   │   ├── __init__.py
│   │   ├── test_model_router.py   # Model selection logic
│   │   ├── test_prompt_classifier.py      # ML classification service
│   │   ├── test_model_registry.py         # Model capability registry
│   │   └── test_yaml_model_loader.py      # YAML configuration loader
│   └── test_main.py               # Main API class tests
└── integration/                   # Integration tests (require running service)
    ├── __init__.py
    ├── test_api_endpoints.py       # HTTP API endpoint testing
    ├── test_routing_consistency.py # Routing behavior consistency
    ├── test_task_routing.py        # Task-specific routing intelligence
    ├── test_model_selection_flows.py       # End-to-end model selection
    ├── test_cost_optimization.py   # Cost optimization behavior
    └── test_partial_model.py       # Partial model specification handling
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Fast, isolated tests that verify individual components without external dependencies.

**Characteristics**:
- No network calls or external service dependencies
- Use mocking for ML models and external services
- Fast execution (< 1 second per test)
- Can run in CI without service dependencies

**Coverage Areas**:
- **Core (`tests/unit/core/`)**: Configuration loading, settings validation, YAML parsing
- **Models (`tests/unit/models/`)**: Pydantic model validation, serialization, enum behavior
- **Services (`tests/unit/services/`)**: Business logic, algorithm correctness, service interactions
- **Main API (`tests/unit/test_main.py`)**: LitServe API wrapper, request/response handling

### Integration Tests (`tests/integration/`)

**Purpose**: End-to-end tests that verify component interactions and real service behavior.

**Characteristics**:
- Requires running adaptive_ai service on `http://localhost:8000`
- Tests actual HTTP endpoints and ML model inference
- Slower execution (1-30 seconds per test)
- Skipped in CI environments unless service is available

**Coverage Areas**:
- **API Endpoints**: HTTP request/response validation, error handling
- **Routing Intelligence**: Task-specific model selection, consistency verification
- **Model Selection Flows**: End-to-end selection with real ML inference
- **Cost Optimization**: Real cost calculation and optimization behavior
- **Partial Model Handling**: Model specification lookup and resolution

## Running Tests

We provide multiple ways to run tests for convenience:

### Using Make Commands (Recommended)
```bash
# Show all available commands
make help

# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage report
make test-cov

# Run with HTML coverage report
make test-cov-html

# Run in fast mode (stop on first failure)
make test-fast

# Run specific test categories
make test-config      # Configuration tests
make test-services    # Service tests
make test-models      # Model tests
make test-routing     # Routing integration tests
```

### Using Shell Script
```bash
# Show all available options
./scripts/test.sh help

# Run all tests
./scripts/test.sh

# Run unit tests only
./scripts/test.sh unit

# Run integration tests only
./scripts/test.sh integration

# Run with coverage
./scripts/test.sh coverage

# Run with HTML coverage
./scripts/test.sh html-cov

# Run specific categories
./scripts/test.sh config
./scripts/test.sh services
./scripts/test.sh models
./scripts/test.sh routing

# Clean test artifacts
./scripts/test.sh clean
```

### Using uv run directly
```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m "unit"

# Integration tests only  
uv run pytest -m "integration"

# Specific test directories
uv run pytest tests/unit/core/
uv run pytest tests/unit/services/
uv run pytest tests/unit/models/
uv run pytest tests/integration/routing/

# With coverage
uv run pytest --cov=adaptive_ai --cov-report=html

# Verbose output
uv run pytest -v

# Fast mode (stop on first failure)
uv run pytest -x --ff
```

### Code Quality Commands
```bash
# Using Make
make lint           # Check code with ruff
make lint-fix       # Fix code issues
make format         # Format with black
make format-check   # Check formatting
make typecheck      # Type checking with mypy
make quality        # Run all quality checks
make quality-fix    # Fix all issues

# Using uv run directly
uv run ruff check .
uv run ruff check --fix .
uv run black .
uv run black --check .
uv run mypy adaptive_ai/
```

## Fixtures

### Core Fixtures (`fixtures/config_fixtures.py`)
- `sample_settings`: Pre-configured Settings instance
- `temp_config_file`: Temporary YAML config file
- `mock_env_vars`: Environment variable mocking

### Model Fixtures (`fixtures/model_fixtures.py`)
- `sample_model_capability`: Pre-built ModelCapability instances
- `sample_model_selection_request`: Request objects for testing
- `sample_classification_result`: Classification result examples

### Request Fixtures (`fixtures/request_fixtures.py`)
- `base_url`: API base URL for integration tests
- `sample_predict_request`: Common prediction requests
- `custom_models_request`: Requests with custom model specs

### ML Fixtures (`fixtures/ml_fixtures.py`)
- `mock_prompt_classifier`: Mocked classification service
- `mock_model_router`: Mocked routing service
- `mock_yaml_model_db`: Mocked model database
- `classification_test_prompts`: Sample prompts for each task type

## Writing New Tests

### Test File Naming
- Unit tests: `test_<component_name>.py`
- Integration tests: `test_<feature_name>.py`
- Keep names descriptive and consistent

### Test Class Organization
```python
class TestComponentName:
    """Test ComponentName class."""
    
    def test_specific_behavior(self):
        """Test specific behavior with descriptive name."""
        # Test implementation
        pass
```

### Markers
- Use `@pytest.mark.unit` for unit tests
- Use `@pytest.mark.integration` for integration tests
- Add custom markers as needed in `pyproject.toml`

### Best Practices
1. **Descriptive test names** - explain what is being tested
2. **Arrange-Act-Assert** pattern
3. **Use fixtures** for common setup
4. **Mock external dependencies** in unit tests
5. **Test edge cases** and error conditions
6. **Keep tests focused** - one concept per test
7. **Use parametrized tests** for multiple similar cases

### Example Test Structure
```python
import pytest
from unittest.mock import patch

from adaptive_ai.services.example import ExampleService

class TestExampleService:
    """Test ExampleService functionality."""
    
    def test_default_behavior(self):
        """Test service with default configuration."""
        service = ExampleService()
        result = service.process()
        assert result.success is True
    
    @pytest.mark.parametrize("input_val,expected", [
        ("test1", "RESULT1"),
        ("test2", "RESULT2"),
    ])
    def test_multiple_inputs(self, input_val, expected):
        """Test service with various inputs."""
        service = ExampleService()
        result = service.process(input_val)
        assert result == expected
    
    def test_error_handling(self):
        """Test service error handling."""
        service = ExampleService()
        with pytest.raises(ValueError):
            service.process(invalid_input=True)
```

## Performance Guidelines

- **Unit tests should run in <100ms** each
- **Integration tests should run in <5s** each  
- **Mock expensive operations** (ML inference, network calls)
- **Use fixtures to share expensive setup** between tests
- **Consider using pytest-benchmark** for performance-critical components

## Debugging Tests

```bash
# Run specific test with output
uv run pytest tests/unit/core/test_config.py::TestSettings::test_from_yaml_simple -v -s

# Run with debugger
uv run pytest tests/unit/core/test_config.py --pdb

# Run last failed tests
uv run pytest --lf

# Run with minimal output
uv run pytest -q
```