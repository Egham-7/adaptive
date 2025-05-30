# Adaptive Python SDK

Python client library for the Adaptive AI platform - drop-in replacement for OpenAI with intelligent model selection.

## Installation

```bash
pip install adaptive-python
```

## Quick Start

```python
from adaptive import Adaptive

# Initialize client
client = Adaptive(api_key="your-api-key")

# Create chat completion (automatically selects optimal model)
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Write a Python function to sort a list"}
    ]
)

print(response.choices[0].message.content)
```

## Features

- **Drop-in OpenAI Replacement**: Compatible with OpenAI Python SDK interface
- **Intelligent Model Selection**: Automatic routing to optimal models
- **Multi-Provider Support**: Access OpenAI, Anthropic, Groq, DeepSeek models
- **Streaming Support**: Real-time response streaming
- **Cost Optimization**: Automatic cost reduction through smart routing
- **Type Safety**: Full TypeScript-style type hints

## Configuration

### Environment Variables

```bash
export ADAPTIVE_API_KEY="your-api-key"
export ADAPTIVE_BASE_URL="https://api.adaptive.ai"  # Optional
```

### Programmatic Configuration

```python
from adaptive import Adaptive

client = Adaptive(
    api_key="your-api-key",
    base_url="https://api.adaptive.ai"  # Optional
)
```

## Usage Examples

### Basic Chat Completion

```python
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Chat Completion

```python
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Error Handling

```python
from adaptive.exceptions import AdaptiveError, AuthenticationError

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
except AdaptiveError as e:
    print(f"API error: {e}")
```

## API Reference

### Chat Completions

#### `client.chat.completions.create()`

**Parameters:**
- `messages` (List[Dict]): List of message objects
- `stream` (bool, optional): Enable streaming responses
- `max_tokens` (int, optional): Maximum tokens in response
- `temperature` (float, optional): Sampling temperature (0-2)

**Returns:**
- `ChatCompletion` object with model response

### Response Objects

```python
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    provider: str  # Added by Adaptive
    choices: List[Choice]
    usage: Usage

class Choice:
    index: int
    message: Message
    finish_reason: str

class Message:
    role: str
    content: str
```

## Migration from OpenAI

Replace OpenAI imports with Adaptive:

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After  
from adaptive import Adaptive
client = Adaptive(api_key="your-adaptive-key")

# Same interface!
response = client.chat.completions.create(...)
```

## Project Structure

```
src/adaptive/
├── __init__.py          # Main client exports
├── adaptive.py          # Core client class
├── resources/           # API resource modules
│   └── chat/           # Chat completions
├── models/             # Response models
├── exceptions/         # Error classes
└── utils/              # Utility functions
```

## Development

```bash
# Install for development
poetry install

# Run tests
poetry run pytest

# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/
```

## Examples

See the `examples/` directory for more usage examples:

- `basic_chat.py` - Simple chat completion
- `streaming_chat.py` - Streaming responses
- `error_handling.py` - Error handling patterns
- `async_usage.py` - Async/await support

## Requirements

- Python 3.12+
- `requests` >= 2.32.3
- `pydantic` >= 2.11.2

## License

MIT License - see LICENSE file for details.