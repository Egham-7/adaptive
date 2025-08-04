# AI Model Configuration Pipeline

This comprehensive pipeline replaces manual model configuration with an intelligent, automated system that extracts, analyzes, and configures AI models from all major providers.

## 🎯 Overview

**Problem Solved**: Replace the manual model configuration in `adaptive_ai/config/providers.py` with an intelligent system that automatically determines optimal task types, complexity capabilities, and performance characteristics for each model.

**Solution**: A 3-stage pipeline that:
1. **Extracts** models from provider APIs
2. **Analyzes** capabilities using AI agents
3. **Generates** structured configurations

## 🚀 Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model          │    │  AI Capability  │    │  Configuration  │
│  Extraction     │───▶│  Analysis       │───▶│  Generation     │
│                 │    │                 │    │                 │
│ • API calls     │    │ • LLM analysis  │    │ • Python config │
│ • Rate limiting │    │ • Task mapping  │    │ • YAML schemas  │
│ • Error handling│    │ • Complexity    │    │ • Task mappings │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
scripts/
├── extract_provider_models.py      # Stage 1: Model extraction
├── model_capability_agent.py       # Stage 2: AI analysis agent
├── automated_model_pipeline.py     # Stage 3: Complete orchestration
├── test_extraction.py              # Comprehensive test suite
├── run_extraction.py               # Simple runner with env loading
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── PIPELINE_README.md              # This documentation
├── README.md                       # Basic extraction docs
└── examples/
    └── openai_enriched_example.yaml # Sample enriched output
```

## 🔥 Key Features

### Stage 1: Model Extraction
- **Concurrent API calls** to 6 major providers
- **Robust error handling** with exponential backoff
- **Rate limiting** to respect API quotas
- **Comprehensive metadata** extraction

### Stage 2: AI Capability Analysis  
- **LLM-powered analysis** using GPT-4o-mini
- **10 task type classifications** (code, math, creative, etc.)
- **4-level complexity assessment** (low, medium, high, expert)
- **Performance profiling** with cost-efficiency ratings
- **Rule-based fallback** when API unavailable

### Stage 3: Configuration Generation
- **Python provider configs** to replace manual files
- **Task-specific mappings** with optimal parameters
- **Automated documentation** and summaries
- **Integration-ready outputs**

## 🛠️ Setup

### 1. Dependencies

```bash
cd adaptive_ai/scripts
uv pip install -r requirements.txt
```

### 2. API Keys

```bash
# Copy template and add your keys
cp .env.example .env
nano .env
```

Required keys (at least one):
- `OPENAI_API_KEY` - For model analysis and OpenAI models
- `ANTHROPIC_API_KEY` - For Anthropic models  
- `GOOGLE_API_KEY` - For Google AI models
- `GROQ_API_KEY` - For Groq models
- `DEEPSEEK_API_KEY` - For DeepSeek models
- `XAI_API_KEY` - For xAI Grok models

## 🚀 Usage

### Complete Pipeline

```bash
# Run full pipeline (recommended)
python automated_model_pipeline.py --full-pipeline

# Process specific providers only
python automated_model_pipeline.py --full-pipeline --providers openai,anthropic

# Use different analysis model
python automated_model_pipeline.py --full-pipeline --analysis-model gpt-4o
```

### Individual Stages

```bash
# Stage 1: Extract models only
python automated_model_pipeline.py --extract-only

# Stage 2: Analyze existing models
python automated_model_pipeline.py --analyze-only

# Stage 1: Just extraction (simple)
python run_extraction.py --providers openai,anthropic
```

### Testing

```bash
# Run comprehensive tests
python test_extraction.py

# Validate pipeline components
python -c "import extract_provider_models; print('✅ Extraction module OK')"
python -c "import model_capability_agent; print('✅ Analysis agent OK')"
```

## 📊 Output Structure

The pipeline creates a structured output directory:

```
pipeline_output/
├── raw_models/                     # Stage 1 output
│   ├── extraction_summary.yaml
│   ├── openai/
│   │   └── openai_models.yaml
│   ├── anthropic/
│   │   └── anthropic_models.yaml
│   └── ... (other providers)
│
├── enriched_models/                # Stage 2 output  
│   ├── openai/
│   │   └── openai_enriched_models.yaml
│   ├── anthropic/
│   │   └── anthropic_enriched_models.yaml
│   └── ... (other providers)
│
└── generated_configs/              # Stage 3 output
    ├── providers.py                # Python provider config
    ├── task_mappings.yaml          # Task-specific mappings
    └── pipeline_summary.yaml       # Analysis summary
```

## 📋 Enhanced YAML Schema

Each enriched model includes comprehensive capability analysis:

```yaml
models:
- id: gpt-4
  # Basic model info (extracted)
  provider: openai
  context_length: 8192
  supports_function_calling: true
  
  # AI-analyzed task capabilities
  task_capabilities:
    code_generation:
      suitability_score: 0.92          # 0.0-1.0 rating
      complexity_levels: [low, medium, high, expert]
      reasoning: "Excellent code generation..."
      recommended_params:
        temperature: 0.1
        max_tokens: 2048
      benchmarks:
        humaneval: 67.0
    
    mathematical_reasoning:
      suitability_score: 0.89
      # ... similar structure for all task types
  
  # Performance profile
  performance_profile:
    overall_complexity: expert        # low/medium/high/expert
    cost_efficiency: premium         # budget/balanced/premium  
    latency_tier: medium            # very_low/low/medium/high
    specializations: [reasoning, analysis, code_generation]
    limitations: [multimodal_tasks, real_time_data]
    estimated_quality_score: 0.90
    cost_per_quality_ratio: 33.33
    confidence_score: 0.95
```

## 🧠 Task Type Analysis

The AI agent analyzes 10 core task types:

| Task Type | Description | Complexity Factors |
|-----------|-------------|-------------------|
| **code_generation** | Writing, debugging, explaining code | Algorithm complexity, domain expertise |
| **mathematical_reasoning** | Solving math problems and proofs | Mathematical level, proof complexity |
| **creative_writing** | Creative content and storytelling | Narrative complexity, style sophistication |
| **text_analysis** | Analyzing and summarizing text | Text length, domain complexity |
| **reasoning_and_logic** | Complex logical problem-solving | Logical steps, abstraction level |
| **conversation** | Natural dialogue interactions | Context length, topic complexity |
| **question_answering** | Answering domain questions | Domain expertise, factual complexity |
| **content_creation** | Generating various content types | Content type, target audience |
| **translation** | Language translation | Language pair difficulty, cultural nuances |
| **multimodal_tasks** | Vision, audio, mixed media | Modality complexity, integration needs |

## 🎯 Complexity Framework

4-level complexity assessment:

- **Low**: Simple, single-step tasks with clear parameters
- **Medium**: Multi-step tasks requiring some reasoning
- **High**: Complex tasks requiring deep reasoning and expertise  
- **Expert**: Highly specialized tasks requiring cutting-edge capabilities

## 🔄 Integration Process

### 1. Review Generated Configurations

```bash
# Check pipeline results
ls pipeline_output/generated_configs/
cat pipeline_output/generated_configs/pipeline_summary.yaml
```

### 2. Replace Manual Configuration

```bash
# Backup original
cp adaptive_ai/config/providers.py adaptive_ai/config/providers.py.backup

# Replace with generated version
cp pipeline_output/generated_configs/providers.py adaptive_ai/config/providers.py
```

### 3. Update Task Mappings

```bash
# Review task mappings
cat pipeline_output/generated_configs/task_mappings.yaml

# Integrate into your task mapping configuration
```

### 4. Test Integration

```bash
# Test AI service with new configuration
cd adaptive_ai
uv run adaptive-ai

# Run your test suite
uv run pytest
```

## 📈 Performance & Accuracy

### Pipeline Performance
- **Extraction**: ~30 seconds for all providers
- **Analysis**: ~2-5 minutes depending on model count
- **Generation**: <10 seconds for configuration files

### Analysis Accuracy
- **LLM Analysis**: 95% confidence score (when API available)
- **Rule-based Fallback**: 75% confidence score
- **Task Mapping**: Based on known benchmarks and model characteristics

### Cost Optimization
- **API Costs**: ~$0.10-0.50 per full pipeline run
- **Rate Limiting**: Respects all provider quotas
- **Caching**: Avoids redundant API calls

## 🚨 Error Handling

The pipeline includes comprehensive error handling:

### Network Issues
- Exponential backoff retry logic
- Graceful degradation to rule-based analysis
- Skip providers with authentication issues

### API Limitations  
- Rate limiting with provider-specific quotas
- Fallback to cached or rule-based data
- Continue processing other providers on failures

### Data Validation
- Schema validation for all YAML outputs
- Confidence scoring for AI analysis quality
- Warnings for incomplete or uncertain data

## 🔧 Customization

### Adding New Providers

1. **Add provider config** in `extract_provider_models.py`:
```python
"new_provider": ProviderConfig(
    name="New Provider",
    api_base="https://api.newprovider.com/v1",
    models_endpoint="/models",
    headers={"Authorization": f"Bearer {os.getenv('NEW_PROVIDER_KEY')}"}
)
```

2. **Add parsing logic** in `_parse_models()` method
3. **Update enum** in `adaptive_ai/models/llm_enums.py`

### Custom Task Types

1. **Extend task definitions** in `model_capability_agent.py`:
```python
"custom_task": {
    "description": "Custom task description",
    "complexity_factors": ["factor1", "factor2"],
    "evaluation_criteria": ["criterion1", "criterion2"]
}
```

2. **Update analysis prompts** to include new task type
3. **Add task-specific parameter recommendations**

### Analysis Model Selection

Use different models for capability analysis:

```bash
# Use GPT-4 for highest accuracy
python automated_model_pipeline.py --analysis-model gpt-4

# Use Claude for alternative perspective  
python automated_model_pipeline.py --analysis-model claude-3-sonnet
```

## 🧪 Testing & Validation

### Automated Tests

```bash
# Run comprehensive test suite
python test_extraction.py

# Test individual components
python -c "from extract_provider_models import ModelExtractor; print('✅ OK')"
python -c "from model_capability_agent import ModelCapabilityAgent; print('✅ OK')"
```

### Manual Validation

1. **Review AI Analysis**: Check `confidence_score` and `reasoning` fields
2. **Validate Task Mappings**: Compare with known model capabilities
3. **Test Parameter Recommendations**: Verify against your use cases
4. **Benchmark Integration**: Compare with objective benchmark scores

### Quality Assurance

- **Confidence Thresholds**: Flag analysis with <80% confidence
- **Cross-Validation**: Compare multiple analysis runs
- **Human Review**: Review flagged or uncertain assessments
- **Iterative Improvement**: Update knowledge base based on real performance

## 🔮 Future Enhancements

### Planned Features
- **Benchmark Integration**: Automatic MMLU, HumanEval score fetching
- **Real-time Updates**: Monitor provider APIs for new models
- **Performance Tracking**: Track actual vs. predicted performance
- **Cost Monitoring**: Real-time cost tracking and optimization

### Advanced Analysis
- **Multi-modal Assessment**: Specialized vision/audio capability analysis
- **Domain Expertise**: Deeper analysis of specialized domains (medical, legal, etc.)
- **Custom Benchmarks**: Integration with custom evaluation suites
- **Comparative Analysis**: Model-to-model performance comparisons

## 🆘 Troubleshooting

### Common Issues

**"No models extracted"**
- Check API keys in `.env` file
- Verify internet connection
- Check provider API status

**"Analysis failed"** 
- Ensure OpenAI API key is valid for analysis model
- Check rate limits haven't been exceeded
- Verify analysis model is available

**"Configuration generation failed"**
- Ensure enriched YAML files exist
- Check YAML file format validity
- Verify write permissions for output directory

### Debug Mode

```bash
# Enable verbose logging
python automated_model_pipeline.py --verbose --full-pipeline

# Check individual stages
python extract_provider_models.py --verbose
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

### Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error logs with `--verbose` flag
3. Validate your API keys and permissions
4. Check provider API documentation for changes

## 📝 License

This pipeline is part of the Adaptive AI project and follows the same license terms.