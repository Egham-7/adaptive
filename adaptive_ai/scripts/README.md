# Provider Model Extraction Scripts

This directory contains scripts to extract all available models from AI providers and organize them into structured YAML files.

## Overview

The extraction script fetches model information from provider APIs and creates organized YAML files for each provider, making it easy to manage and update model configurations.

## Features

- ✅ **Async Operations**: Concurrent API calls for fast extraction
- ✅ **Comprehensive Coverage**: Supports all major AI providers
- ✅ **Error Handling**: Robust retry logic and rate limiting
- ✅ **Structured Output**: Clean YAML files with metadata
- ✅ **Selective Extraction**: Filter specific providers
- ✅ **Environment Management**: Secure API key handling

## Supported Providers

| Provider | API Endpoint | Models Extracted |
|----------|-------------|------------------|
| OpenAI | `api.openai.com` | GPT-4, GPT-3.5, embeddings |
| Anthropic | `api.anthropic.com` | Claude models |
| Google AI | `generativelanguage.googleapis.com` | Gemini models |
| Groq | `api.groq.com` | Llama, Mixtral models |
| DeepSeek | `api.deepseek.com` | DeepSeek models |
| xAI Grok | `api.x.ai` | Grok models |

## Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
cd adaptive_ai/scripts
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Add your API keys to the `.env` file:

```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
GROQ_API_KEY=your-groq-key
DEEPSEEK_API_KEY=your-deepseek-key
XAI_API_KEY=your-xai-grok-key
```

## Usage

### Extract All Provider Models

```bash
# Using the simple runner (loads .env automatically)
python run_extraction.py

# Or using the main script directly
python extract_provider_models.py
```

### Extract Specific Providers

```bash
# Extract only OpenAI and Anthropic models
python run_extraction.py --providers openai,anthropic

# Extract with custom output directory
python run_extraction.py --output-dir my_models

# Enable verbose logging
python run_extraction.py --verbose
```

### Command Line Options

```bash
python extract_provider_models.py --help

Options:
  --providers TEXT    Comma-separated list of providers (e.g., 'openai,anthropic')
  --output-dir TEXT   Output directory for YAML files (default: 'models')
  --verbose          Enable verbose logging
  --help             Show help message
```

## Output Structure

The script creates the following directory structure:

```
models/
├── extraction_summary.yaml          # Overall summary
├── openai/
│   └── openai_models.yaml          # OpenAI models
├── anthropic/
│   └── anthropic_models.yaml       # Anthropic models
├── google/
│   └── google_models.yaml          # Google AI models
└── ... (other providers)
```

### YAML File Format

Each provider's YAML file contains:

```yaml
provider:
  name: "OpenAI"
  id: "openai"
  extracted_at: "2024-01-15T10:30:00"
  total_models: 15

models:
  - id: "gpt-4"
    name: "gpt-4"
    provider: "openai"
    type: "model"
    max_tokens: 4096
    context_length: 8192
    supports_function_calling: true
    supports_vision: false
    supports_streaming: true
    created: "2023-03-01"
    owned_by: "openai"
  # ... more models
```

### Summary File

The `extraction_summary.yaml` provides an overview:

```yaml
extraction_summary:
  extracted_at: "2024-01-15T10:30:00"
  total_providers: 8
  total_models: 150

providers:
  openai:
    name: "OpenAI"
    total_models: 15
    models: ["gpt-4", "gpt-3.5-turbo", "..."]
    has_more: true
  # ... other providers
```

## Model Information Extracted

For each model, the script attempts to extract:

- **Basic Info**: ID, name, provider, type
- **Capabilities**: Function calling, vision, streaming support
- **Limits**: Max tokens, context length
- **Metadata**: Creation date, owner, description
- **Pricing**: Input/output costs (when available)

## Error Handling

The script includes robust error handling:

- **Authentication Errors**: Skips providers with invalid API keys
- **Rate Limiting**: Respects API quotas with exponential backoff
- **Network Issues**: Retries failed requests up to 3 times
- **Parsing Errors**: Logs errors and continues with other providers

## Example Output

```bash
$ python run_extraction.py --providers openai,anthropic

2024-01-15 10:30:00 - INFO - Starting model extraction...
2024-01-15 10:30:01 - INFO - Fetching models from OpenAI...
2024-01-15 10:30:02 - INFO - Successfully parsed 15 models from OpenAI
2024-01-15 10:30:02 - INFO - Fetching models from Anthropic...
2024-01-15 10:30:03 - INFO - Successfully parsed 8 models from Anthropic
2024-01-15 10:30:03 - INFO - Saving models to YAML files...
2024-01-15 10:30:03 - INFO - Saved 15 models for openai to models/openai/openai_models.yaml
2024-01-15 10:30:03 - INFO - Saved 8 models for anthropic to models/anthropic/anthropic_models.yaml
2024-01-15 10:30:03 - INFO - Created extraction summary: models/extraction_summary.yaml
2024-01-15 10:30:03 - INFO - ✅ Extraction complete! Found 23 models across 2 providers
```

## Integration with Adaptive AI

The extracted YAML files can be used to:

1. **Update Provider Configurations**: Import model metadata into `config/providers.py`
2. **Model Registry**: Populate the model registry with comprehensive model data
3. **Cost Analysis**: Use pricing information for cost optimization
4. **Capability Mapping**: Map model capabilities to task requirements

## Best Practices

### Security
- Never commit API keys to version control
- Use environment variables or `.env` files for sensitive data
- Rotate API keys regularly

### Performance
- Use `--providers` to limit extraction to needed providers
- Run extraction during off-peak hours to respect rate limits
- Cache results locally to avoid repeated API calls

### Maintenance
- Re-run extraction regularly to catch new models
- Monitor provider API changes that might break parsing
- Update model capability mappings as providers add features

## Troubleshooting

### Common Issues

**Authentication Failed**
```
ERROR - OpenAI: Authentication failed
```
- Check your API key in `.env`
- Verify the key has necessary permissions
- Ensure no extra spaces in the key value

**Rate Limited**
```
WARNING - Anthropic: Rate limited
```
- The script will automatically retry with backoff
- Consider running during off-peak hours
- Use `--providers` to reduce concurrent requests

**No Models Found**
```
WARNING - No models found for provider, skipping YAML creation
```
- Check API key permissions
- Verify provider API endpoint is accessible
- Some providers require special access for model listing

**Network Errors**
```
ERROR - Error fetching models from Provider: Connection timeout
```
- Check internet connection
- Verify firewall settings
- Some corporate networks block AI provider APIs

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python run_extraction.py --verbose
```

This will show:
- Detailed HTTP requests and responses
- Model parsing steps
- Cache operations
- Full error stack traces

## Contributing

To add support for new providers:

1. Add provider configuration to `_get_provider_configs()`
2. Implement provider-specific parsing in `_parse_models()`
3. Add capability detection methods
4. Update this README with provider information
5. Test with actual API credentials

## License

This script is part of the Adaptive AI project and follows the same license terms.