# Test Suite Documentation

Test suite for Adaptive Router - intelligent LLM model selection library.

## Test Structure

```
tests/
├── README.md                      # This documentation
├── conftest.py                    # Global pytest configuration
├── fixtures/                      # Reusable test fixtures
│   ├── config_fixtures.py         # Configuration and settings fixtures
│   └── ml_fixtures.py             # ML model and service mocks
├── unit/                          # Unit tests (fast, isolated)
│   ├── core/                      # Core functionality tests
│   │   └── test_config.py         # Configuration loading
│   ├── models/                    # Data model tests
│   │   ├── test_core_models.py    # Core request/response models
│   │   └── test_classification_models.py  # Classification models
│   └── services/                  # Service layer tests
│       ├── test_model_router.py   # Router logic
│       ├── test_prompt_classifier.py  # ML classifier
│       ├── test_registry_service.py   # Registry cache backed by HTTP client
│       └── test_yaml_model_loader.py  # YAML loader
└── integration/                   # Integration tests (require services)
    └── test_api_endpoints.py      # End-to-end API tests
```

## Test Categories

### Unit Tests

**Location**: `tests/unit/`
**Marker**: `@pytest.mark.unit`
**Dependencies**: None (all mocked)

Fast tests that verify individual components:
- Model validation
- Service logic
- Configuration parsing
- Data transformations

**Run**:
```bash
uv run pytest -m unit
```

### Integration Tests

**Location**: `tests/integration/`
**Marker**: `@pytest.mark.integration`
**Dependencies**: Running services (FastAPI server)

Tests that verify end-to-end workflows:
- HTTP API endpoints
- Library usage patterns
- Request/response validation

**Run**:
```bash
# Most integration tests are skipped by default
uv run pytest -m integration

# To run them, you need services running:
# Terminal 1: uv run adaptive-router
# Terminal 2: uv run pytest -m integration --no-skip
```

## Running Tests

### Basic Commands

```bash
# All tests
uv run pytest

# Unit tests only (fast, no deps)
uv run pytest -m unit

# Integration tests (requires services)
uv run pytest -m integration

# Specific file
uv run pytest tests/unit/services/test_model_router.py

# Specific test
uv run pytest tests/unit/services/test_model_router.py::TestModelRouter::test_initialization

# Verbose
uv run pytest -v

# With coverage
uv run pytest --cov
uv run pytest --cov --cov-report=html
```

### Make Commands

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-cov          # With coverage
make test-cov-html     # HTML coverage report
```

## Test Fixtures

### Config Fixtures

**File**: `fixtures/config_fixtures.py`

```python
@pytest.fixture
def temp_yaml_config() -> Generator[str, None, None]:
    """Create temporary YAML config file."""

@pytest.fixture
def test_settings() -> Settings:
    """Pre-configured test settings."""
```

### ML Fixtures

**File**: `fixtures/ml_fixtures.py`

Mock ML models and classifiers for testing without loading actual models.

## Writing Tests

### Unit Test Template

```python
import pytest
from adaptive_router.services.model_router import ModelRouter
from adaptive_router.services.model_registry import ModelRegistry
from adaptive_router.services.yaml_model_loader import YAMLModelDatabase

@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create test registry."""
    yaml_db = YAMLModelDatabase()
    return ModelRegistry(yaml_db)

@pytest.mark.unit
def test_feature(model_registry: ModelRegistry) -> None:
    """Test feature description."""
    # Arrange
    router = ModelRouter(model_registry)

    # Act
    result = router._select_models(
        task_complexity=0.5,
        task_type="general",
        cost_bias=0.5
    )

    # Assert
    assert len(result) > 0
```

### Integration Test Template

```python
import pytest

@pytest.mark.integration
@pytest.mark.skip(reason="Requires FastAPI server on localhost:8000")
def test_api_endpoint() -> None:
    """Test API endpoint."""
    import requests

    response = requests.post(
        "http://localhost:8000/select_model",
        json={"prompt": "Test prompt", "cost_bias": 0.5}
    )

    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "model" in data
```

## Test Coverage

### Running Coverage

```bash
# Terminal report
uv run pytest --cov --cov-report=term-missing

# HTML report
uv run pytest --cov --cov-report=html
open htmlcov/index.html
```

### Coverage Goals

- Overall: >80%
- Services: >90%
- Models: >95%

### Coverage Configuration

See `pyproject.toml` for exclusions:
- Debug code
- Abstract methods
- Type checking blocks
- `if __name__ == "__main__"`

## Continuous Integration

CI runs on pull requests and commits:

**Always Run**:
- Unit tests (`pytest -m unit`)
- Linting (`ruff check`)
- Type checking (`mypy`)
- Formatting check (`black --check`)

**Skipped**:
- Integration tests (require services)

## Debugging Tests

### Verbose Output

```bash
uv run pytest -vv
uv run pytest -vv -s  # Don't capture stdout
```

### Run Single Test

```bash
uv run pytest tests/unit/services/test_model_router.py::TestModelRouter::test_initialization -vv
```

### Print Debugging

```python
def test_feature():
    result = some_function()
    print(f"Result: {result}")  # Shows with -s flag
    assert result is not None
```

### PDB Debugging

```python
def test_feature():
    result = some_function()
    import pdb; pdb.set_trace()  # Breakpoint
    assert result is not None
```

Run with:
```bash
uv run pytest tests/unit/services/test_model_router.py -s
```

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'adaptive_router'`

**Solution**:
```bash
uv install -e .
```

### Slow First Run

**Problem**: First test run is slow

**Solution**: This is normal. The first run downloads ML models (~500MB). Cached in `~/.cache/huggingface/`. Subsequent runs are faster.

### Test Isolation

**Problem**: Tests fail when run together but pass individually

**Solution**: Tests aren't properly isolated. Check for:
- Global state
- Shared fixtures
- File system dependencies

Use fixtures properly:
```python
@pytest.fixture
def isolated_resource():
    """Creates fresh resource per test."""
    resource = create_resource()
    yield resource
    cleanup_resource(resource)
```

## Best Practices

1. **One Assertion Per Test**: Focus each test
2. **Descriptive Names**: `test_router_selects_cheapest_model_with_zero_cost_bias`
3. **AAA Pattern**: Arrange, Act, Assert
4. **Use Fixtures**: Don't repeat setup code
5. **Mock External Deps**: Keep unit tests fast
6. **Mark Appropriately**: Use `@pytest.mark.unit` or `@pytest.mark.integration`
7. **Document Complex Tests**: Add docstrings explaining "why"

## Test Markers

Available markers (see `pyproject.toml`):

```python
@pytest.mark.unit          # Fast, isolated tests
@pytest.mark.integration   # Require services
@pytest.mark.asyncio       # Async tests
```

Filter by marker:
```bash
uv run pytest -m unit
uv run pytest -m "unit and not slow"
```

## Adding New Tests

1. **Choose category**: Unit or integration
2. **Create test file**: `test_<feature>.py`
3. **Import fixtures**: Use existing or create new
4. **Write tests**: Follow AAA pattern
5. **Add markers**: `@pytest.mark.unit` etc.
6. **Run tests**: `uv run pytest`
7. **Check coverage**: `uv run pytest --cov`
8. **Update docs**: Add to this README if needed
