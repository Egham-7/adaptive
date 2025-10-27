# Adaptive Router Package

Core Python package for intelligent LLM model selection.

## Package Structure

```
adaptive_router/
├── __init__.py              # Public API exports
├── models/                  # Pydantic data models
│   ├── llm_core_models.py          # Request/response models
│   └── llm_classification_models.py # Classification results
├── services/                # Business logic
│   ├── model_router.py              # Main router (public API)
│   ├── model_registry.py            # Model metadata management
│   ├── yaml_model_loader.py         # YAML config loader
│   └── prompt_task_complexity_classifier.py  # ML classifier
└── utils/                   # Utilities
    └── jwt.py                       # JWT authentication
```

## Public API

The package exports the following for library usage:

### Core Classes

```python
from adaptive_router import (
    ModelRouter,              # Main entry point
    ModelSelectionRequest,    # Request model
    ModelSelectionResponse,   # Response model
    ModelCapability,          # Model metadata
    PromptClassifier,        # ML classifier
    YAMLModelDatabase,       # Model database
    ModelRegistry,           # Model registry
)
```

### Main Usage

```python
from adaptive_router import ModelRouter, ModelSelectionRequest

# Initialize (handles all dependencies)
router = ModelRouter()

# Make selection
request = ModelSelectionRequest(prompt="Your prompt here")
response = router.select_model(request)
```

## Services

### ModelRouter

**File**: `services/model_router.py`

Main public API. Orchestrates classification and model selection.

**Public Method**:
- `select_model(request: ModelSelectionRequest) -> ModelSelectionResponse`

**Internal Dependencies** (auto-initialized):
- `PromptClassifier`: ML-based prompt analysis
- `ModelRegistry`: Model metadata and filtering
- `YAMLModelDatabase`: YAML-based model config

### PromptClassifier

**File**: `services/prompt_task_complexity_classifier.py`

NVIDIA DeBERTa-based ML classifier for prompt analysis.

**Features**:
- Task type classification
- Complexity scoring (0.0-1.0)
- Domain knowledge assessment
- Direct GPU/CPU inference

**Usage**:
```python
from adaptive_router import PromptClassifier

classifier = PromptClassifier()
result = classifier.classify_prompt("Write a sorting algorithm")
# Returns dict with task_type_1, prompt_complexity_score, etc.
```

### ModelRegistry

**File**: `services/model_registry.py`

Manages model metadata and capabilities.

**Features**:
- Model lookup by unique_id (provider:model_name)
- Model filtering by criteria
- Provider availability checking

**Usage**:
```python
from adaptive_router import ModelRegistry, YAMLModelDatabase

yaml_db = YAMLModelDatabase()
registry = ModelRegistry(yaml_db)

# Get model
model = registry.get_model_capability("openai:gpt-4")

# Find models by criteria
models = registry.find_models_matching_criteria(
    ModelCapability(provider="openai", max_context_tokens=8000)
)
```

### YAMLModelDatabase

**File**: `services/yaml_model_loader.py`

Loads model definitions from YAML files.

**Features**:
- Parses YAML model configs
- Validates model metadata
- Provides fast in-memory lookup

**YAML Format**:
```yaml
models:
  gpt-4:
    model_name: "gpt-4"
    cost_per_1m_input_tokens: 30.0
    cost_per_1m_output_tokens: 60.0
    max_context_tokens: 8192
    supports_function_calling: true
```

## Models

### Core Models

**File**: `models/llm_core_models.py`

- `ModelSelectionRequest`: Input to select_model()
- `ModelSelectionResponse`: Output from select_model()
- `ModelCapability`: Model metadata and capabilities
- `Alternative`: Alternative model recommendation

## Design Principles

1. **Single Entry Point**: `ModelRouter` is the only class users need
2. **Automatic Dependencies**: All internal services are auto-initialized
3. **Immutable Responses**: All models use Pydantic for validation
4. **Type Safety**: Full type hints and mypy compliance
5. **Privacy**: Internal methods prefixed with `_`

## Extension Points

To add new providers:

1. Add YAML file to `../model_data/data/provider_models/`
2. Follow naming: `{provider}_models_structured.yaml`
3. Restart to reload model database

To customize classification:

1. Subclass `PromptClassifier`
2. Override `classify_prompt()` method
3. Pass to `ModelRouter(prompt_classifier=custom_classifier)`

## Testing

Unit tests in `../tests/unit/`:
- `test_model_router.py`: Router logic tests
- `test_model_registry.py`: Registry tests
- `test_prompt_classifier.py`: Classifier tests

Integration tests in `../tests/integration/`:
- `test_api_endpoints.py`: End-to-end tests
