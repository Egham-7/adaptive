# Helicone AI Models Database - Structured YAML Files

This directory contains beautifully structured YAML files with live AI model pricing data from Helicone's API.

## 🌟 Features

- **Live Data**: Fetched directly from Helicone's production API
- **Structured Format**: Models organized by categories with clear hierarchy
- **Comprehensive Coverage**: 985+ models across 20+ providers
- **Rich Metadata**: Includes cache pricing, audio costs, batch pricing
- **Ready for ML**: Empty capability fields for manual AI classification
- **Professional Formatting**: Clean YAML with extensive documentation

## 📁 File Structure

```
structured_provider_models/
├── 00_master_index.yaml           # Overview of all providers
├── README.md                      # This documentation
├── anthropic_models_structured.yaml
├── avian_models_structured.yaml
├── aws_models_structured.yaml
├── azure_models_structured.yaml
├── cohere_models_structured.yaml
├── deepseek_models_structured.yaml
├── fireworks_models_structured.yaml
├── google_models_structured.yaml
├── groq_models_structured.yaml
├── llama_models_structured.yaml
├── mistral_models_structured.yaml
├── nebius_models_structured.yaml
├── novita_models_structured.yaml
├── openai_models_structured.yaml
├── openrouter_models_structured.yaml
├── perplexity_models_structured.yaml
├── qstash_models_structured.yaml
├── together_models_structured.yaml
├── vercel_models_structured.yaml
├── x_models_structured.yaml
```

## 📊 Provider Statistics

| Provider | Models | Key Model Examples |
|----------|--------|-------------------|
| OPENROUTER | 312 | 01-ai/yi-large, aetherwiing/mn-starcannon-12b, +310 more |
| AZURE | 147 | ada, ada-batch, +145 more |
| OPENAI | 140 | ada, ada-batch, +138 more |
| TOGETHER | 116 | allenai/OLMo-7B, allenai/OLMo-7B-Instruct, +114 more |
| VERCEL | 99 | alibaba/qwen-3-14b, alibaba/qwen-3-235b, +97 more |
| NOVITA | 35 | cognitivecomputations/dolphin-mixtral-8x22b, deepseek/deepseek-r1, +33 more |
| MISTRAL | 24 | codestral, devstral-medium, +22 more |
| FIREWORKS | 23 | accounts/fireworks/models/deepseek-v3, accounts/fireworks/models/gpt-oss-120b, +21 more |
| GOOGLE | 20 | claude-3-5-haiku, claude-3-5-sonnet, +18 more |
| GROQ | 20 | deepseek-r1-distill-llama-70b, gemma-7b-it, +18 more |
| ANTHROPIC | 15 | claude-2, claude-2.0, +13 more |
| X | 9 | grok-2-1212, grok-2-vision-1212, +7 more |
| LLAMA | 7 | Cerebras-Llama-4-Maverick-17B-128E-Instruct, Cerebras-Llama-4-Scout-17B-16E-Instruct, +5 more |
| AVIAN | 4 | Meta-Llama-3.1-405B-Instruct, Meta-Llama-3.1-70B-Instruct, +2 more |
| AWS | 4 | amazon.nova-lite-v1%3A0, amazon.nova-micro-v1%3A0, +2 more |
| NEBIUS | 3 | black-forest-labs/flux-dev, black-forest-labs/flux-schnell, +1 more |
| PERPLEXITY | 3 | sonar, sonar-pro, +1 more |
| QSTASH | 2 | llama, mistral |
| COHERE | 1 | cohere/command-r |
| DEEPSEEK | 1 | deepseek-chat |

**Total: 985 models across 20 providers**

## 🎯 YAML Structure

Each provider YAML file follows this simple structure:

```yaml
provider_info:
  name: OPENAI
  total_models: 140
  data_source: https://helicone.ai/api/llm-costs?provider=openai
  last_updated: 2025-01-06
  currency: USD
  pricing_unit: per 1 million tokens

models:
  gpt_4o:
      description: ''        # Empty for manual assignment
      provider: OPENAI
      model_name: gpt-4o
      cost_per_1m_input_tokens: 2.5
      cost_per_1m_output_tokens: 10.0
      max_context_tokens: null    # Empty for manual assignment
      max_output_tokens: null     # Empty for manual assignment
      supports_function_calling: null  # Empty for manual assignment
      languages_supported: []     # Empty for manual assignment
      model_size_params: ''       # Empty for manual assignment
      latency_tier: ''           # Empty for manual assignment
      task_type: ''              # Empty from beginning
      complexity: ''             # Empty from beginning
      additional_pricing:        # Optional, if available
        cache_read_per_1m: 0.5
        cache_write_per_1m: 1.25
      metadata:
        matching_operator: equals
        available_in_playground: true
```

## 🚀 Usage Examples

### Load All Provider Data
```python
import yaml
from pathlib import Path

# Load master index
with open('00_master_index.yaml') as f:
    index = yaml.safe_load(f)

# Load specific provider
with open('openai_models_structured.yaml') as f:
    openai_models = yaml.safe_load(f)
```

### Extract Pricing Data
```python
# Get specific model and its pricing
gpt4o_model = openai_models['models']['gpt_4o']
input_cost = gpt4o_model['cost_per_1m_input_tokens']
output_cost = gpt4o_model['cost_per_1m_output_tokens']
print(f'GPT-4o: ${input_cost}/${output_cost} per 1M tokens')
```

## 🔄 Data Freshness

- **Source**: Live Helicone API (not cached or hardcoded)
- **Update Frequency**: Run the generation script to get latest data
- **Accuracy**: Directly from Helicone's production cost database

## 🎨 Why This Structure?

1. **Simple Organization**: Each model as a separate entry for easy access
2. **ModelCapability Format**: Follows the exact structure needed for AI routing
3. **Extensible Fields**: Ready for manual assignment of capabilities and metadata
4. **Rich Pricing Data**: Includes all available pricing dimensions from Helicone
5. **Production Ready**: Clean, documented, and maintainable YAML format

*Generated on 2025-08-06 20:38:43 from Helicone API*
