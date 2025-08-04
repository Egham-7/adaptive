# Implementation Steps: Automated Model Configuration Pipeline

This document provides step-by-step instructions to implement the intelligent model configuration system that replaces your manual `adaptive_ai/config/providers.py` setup.

## 🎯 Overview

You'll replace the current manual model configuration with an AI-powered pipeline that:
- **Extracts** all models from provider APIs
- **Analyzes** capabilities using AI agents (matching your TaskType enum)
- **Generates** structured configurations compatible with your existing system

## 📋 Prerequisites

### 1. API Keys Setup

You need at least **one** API key, but more providers = better coverage:

```bash
# Navigate to scripts directory
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai/scripts

# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required Keys (get at least OpenAI + 1-2 others):**
```bash
# Essential for AI analysis
OPENAI_API_KEY=sk-your-openai-key-here

# Provider keys (get what you have access to)
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-ai-key
GROQ_API_KEY=your-groq-key
DEEPSEEK_API_KEY=your-deepseek-key
XAI_API_KEY=your-xai-grok-key
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai/scripts
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 3. Test Installation

```bash
# Test the pipeline components
python test_extraction.py
```

You should see:
```
🎉 All tests passed! The extraction script is working correctly.
```

## 🚀 Step-by-Step Implementation

### Step 1: Backup Current Configuration

```bash
# Backup your current manual configuration
cp /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py \
   /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py.backup

echo "✅ Backed up current providers.py"
```

### Step 2: Run the Complete Pipeline

```bash
# Run full automated pipeline
python automated_model_pipeline.py --full-pipeline

# Or if you want to limit to specific providers (recommended for first run)
python automated_model_pipeline.py --full-pipeline --providers openai,anthropic,deepseek
```

**Expected Output:**
```
🚀 Starting full model configuration pipeline...
============================================================
Running: Extract Models
============================================================
🚀 Step 1: Extracting models from provider APIs...
✅ Extracted 25 models from 3 providers

============================================================
Running: Analyze Capabilities  
============================================================
🧠 Step 2: Analyzing model capabilities...
✅ Model capability analysis complete

============================================================
Running: Generate Configurations
============================================================
⚙️ Step 3: Generating configuration files...
✅ Configuration generation complete

🎉 Pipeline completed successfully!
```

### Step 3: Review Generated Results

```bash
# Check the pipeline output structure
ls -la pipeline_output/

# Review the summary
cat pipeline_output/generated_configs/pipeline_summary.yaml

# Check generated provider config
head -50 pipeline_output/generated_configs/providers.py
```

**Expected Structure:**
```
pipeline_output/
├── raw_models/              # API extracted data
├── enriched_models/         # AI-analyzed capabilities  
└── generated_configs/       # Ready-to-use configurations
    ├── providers.py         # 🎯 Main replacement file
    ├── task_mappings.yaml   # Task-specific mappings
    └── pipeline_summary.yaml # Analysis summary
```

### Step 4: Validate Generated Configuration

```bash
# Test the generated configuration syntax
python -c "
import sys
sys.path.append('pipeline_output/generated_configs')
try:
    import providers
    print('✅ Generated providers.py is valid Python')
    print(f'📊 Found {len(providers.provider_model_capabilities)} providers')
    for provider, models in providers.provider_model_capabilities.items():
        print(f'   {provider}: {len(models)} models')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### Step 5: Review AI Analysis Quality

```bash
# Check AI analysis confidence scores
python -c "
import yaml
import os
from pathlib import Path

confidence_scores = []
provider_dirs = Path('pipeline_output/enriched_models').iterdir()

for provider_dir in provider_dirs:
    if provider_dir.is_dir():
        yaml_files = list(provider_dir.glob('*_enriched_models.yaml'))
        if yaml_files:
            with open(yaml_files[0]) as f:
                data = yaml.safe_load(f)
            
            for model in data.get('models', []):
                profile = model.get('performance_profile', {})
                confidence = profile.get('confidence_score', 0)
                if confidence > 0:
                    confidence_scores.append((model['id'], confidence))

if confidence_scores:
    avg_confidence = sum(score for _, score in confidence_scores) / len(confidence_scores)
    print(f'📊 Average AI analysis confidence: {avg_confidence:.2f}')
    
    low_confidence = [(model, score) for model, score in confidence_scores if score < 0.8]
    if low_confidence:
        print(f'⚠️  Models with low confidence (<0.8):')
        for model, score in low_confidence:
            print(f'   {model}: {score:.2f}')
    else:
        print('✅ All models have high confidence scores (>0.8)')
else:
    print('❌ No confidence scores found')
"
```

### Step 6: Replace Current Configuration

**⚠️ IMPORTANT: Only proceed if Step 5 shows good confidence scores**

```bash
# Replace the current providers.py with generated version
cp pipeline_output/generated_configs/providers.py \
   /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py

echo "✅ Replaced providers.py with AI-generated configuration"
```

### Step 7: Test Integration

```bash
# Test that the AI service starts with new configuration
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai

# Check for syntax errors
python -c "
try:
    from adaptive_ai.config.providers import provider_model_capabilities
    print('✅ New providers.py imports successfully')
    
    # Count models per provider
    for provider, models in provider_model_capabilities.items():
        print(f'   {provider}: {len(models)} models')
        
    total_models = sum(len(models) for models in provider_model_capabilities.values())
    print(f'📊 Total models configured: {total_models}')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    print('   You may need to restore the backup and check the generated file')
"
```

### Step 8: Test AI Service

```bash
# Start the AI service to test integration
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai
uv run adaptive-ai
```

**Expected:** Service should start without errors and show the new model configurations.

### Step 9: Run Tests

```bash
# Run your existing test suite
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai
uv run pytest

# Check specific tests related to model selection
uv run pytest tests/ -v -k "model" || echo "No model-specific tests found"
```

### Step 10: Update Task Mappings (Optional)

```bash
# Review generated task mappings
cat pipeline_output/generated_configs/task_mappings.yaml

# Compare with current task mappings
cat /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/task_mappings.py
```

**Manual Integration:** The task mappings are more complex and may need manual integration based on your specific use case.

## 🔧 Customization & Maintenance

### Regular Updates

```bash
# Create a simple update script
cat > update_models.sh << 'EOF'
#!/bin/bash
echo "🔄 Updating model configurations..."

cd /Users/attaimen/gitrepos/adaptive/adaptive_ai/scripts

# Backup current config
cp ../adaptive_ai/config/providers.py ../adaptive_ai/config/providers.py.backup.$(date +%Y%m%d)

# Run pipeline
python automated_model_pipeline.py --full-pipeline

# Replace configuration
cp pipeline_output/generated_configs/providers.py ../adaptive_ai/config/providers.py

echo "✅ Model configuration updated!"
EOF

chmod +x update_models.sh
```

### Adding New Providers

When new providers become available:

1. **Add provider config** in `extract_provider_models.py`
2. **Add to enum** in `adaptive_ai/models/llm_enums.py`
3. **Run pipeline** to automatically include new models

### Custom Analysis

```bash
# Use different analysis model for better/cheaper analysis
python automated_model_pipeline.py --full-pipeline --analysis-model gpt-4

# Process only specific providers
python automated_model_pipeline.py --full-pipeline --providers openai,anthropic
```

## 🆘 Troubleshooting

### Common Issues

**1. "No models extracted"**
```bash
# Check API keys
cat .env | grep -v "your_" | grep "_API_KEY="

# Test individual provider
python extract_provider_models.py --providers openai --verbose
```

**2. "Analysis failed"**
```bash
# Check OpenAI API key (required for analysis)
python -c "
import os
key = os.getenv('OPENAI_API_KEY', '')
if key and not key.endswith('your_'):
    print('✅ OpenAI key configured')
else:
    print('❌ OpenAI key missing or placeholder')
"

# Use rule-based fallback
python automated_model_pipeline.py --full-pipeline --analysis-model none
```

**3. "Import errors after replacement"**
```bash
# Restore backup
cp /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py.backup \
   /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py

# Check generated file
python -m py_compile pipeline_output/generated_configs/providers.py
```

**4. "Service won't start"**
```bash
# Check configuration syntax
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai
python -c "from adaptive_ai.config.providers import provider_model_capabilities; print('OK')"

# Check for missing imports
python -c "
try:
    from adaptive_ai.models.llm_core_models import ModelCapability
    from adaptive_ai.models.llm_enums import ProviderType
    print('✅ Required imports available')
except ImportError as e:
    print(f'❌ Missing import: {e}')
"
```

### Recovery Steps

If something goes wrong:

```bash
# 1. Restore backup
cp /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py.backup \
   /Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/providers.py

# 2. Test service
cd /Users/attaimen/gitrepos/adaptive/adaptive_ai
uv run adaptive-ai

# 3. Debug and re-run pipeline
cd scripts
python automated_model_pipeline.py --verbose --full-pipeline
```

## 📊 Validation Checklist

Before going to production, verify:

- [ ] **Pipeline completes successfully** without errors
- [ ] **High confidence scores** (>0.8 average) in AI analysis
- [ ] **Generated providers.py** imports without syntax errors
- [ ] **AI service starts** with new configuration
- [ ] **Test suite passes** with new configuration
- [ ] **Task mappings align** with your expected task types
- [ ] **Model coverage** includes your key providers
- [ ] **Backup created** of original configuration

## 🎯 Success Metrics

After implementation, you should have:

- **Automated Discovery**: New models automatically included
- **Intelligent Mapping**: AI-determined task suitability scores
- **Consistent Structure**: All 11 TaskType enum values covered
- **Performance Optimization**: Models ranked by capability and cost
- **Easy Maintenance**: Simple re-run to update configurations

## 🔄 Next Steps

1. **Schedule Regular Updates**: Run monthly to capture new models
2. **Monitor Performance**: Track actual vs. predicted model performance
3. **Refine Analysis**: Improve AI analysis based on real usage data
4. **Expand Integration**: Use enriched data for better cost optimization

## 📞 Support

If you encounter issues:

1. **Check logs** with `--verbose` flag
2. **Review confidence scores** for analysis quality
3. **Test individual components** (extraction, analysis, generation)
4. **Use backup** to quickly restore if needed

The pipeline is designed to be robust and recoverable - you can always restore your backup and retry with different parameters.