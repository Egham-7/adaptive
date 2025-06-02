# Adaptive AI Model Selection

An intelligent system for selecting the most appropriate AI model based on prompt analysis and task requirements.

## Features

- Dynamic model selection based on prompt complexity and task type
- Support for multiple AI providers (OpenAI, Anthropic, GROQ)
- Domain-specific complexity analysis
- Caching for improved performance
- Configurable through environment variables or YAML
- Comprehensive error handling and logging
- Type-safe implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive.git
cd adaptive/adaptive_ai
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure the application:
   - Copy `config.yaml.example` to `config.yaml`
   - Update the configuration as needed
   - Set environment variables if required

## Usage

1. Start the API server:
```bash
python main.py
```

2. Make requests to the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "domain": "Computers_and_Electronics"}'
```

## Configuration

The application can be configured through:

1. Environment variables:
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export DEFAULT_MODEL=gpt-4-turbo
```

2. YAML configuration file (`config.yaml`):
```yaml
api_host: "0.0.0.0"
api_port: 8000
default_model: "gpt-4-turbo"
```

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
ruff check .
```

4. Run type checking:
```bash
mypy .
```

## Project Structure

```
adaptive_ai/
├── main.py              # API entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
├── core/               # Core functionality
│   └── config.py       # Configuration management
├── models/             # Model definitions
│   └── llms.py         # LLM configurations
├── services/           # Business logic
│   ├── model_selector.py    # Model selection logic
│   └── prompt_classifier.py # Prompt analysis
└── tests/              # Test suite
    └── test_model_selector.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
