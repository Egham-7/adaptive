# Model Data & Enrichment Tools

AI-powered model metadata enrichment using LangGraph workflow and GPT-4o-mini.

## Quick Start

### 1. Get Model Data
The `data/provider_models/` directory contains YAML files with metadata for 20+ providers:

```bash
ls data/provider_models/
# anthropic_models_structured.yaml
# openai_models_structured.yaml  
# groq_models_structured.yaml
# ... (20 provider files)
```

### 2. Run AI Enrichment
Set your OpenAI API key and run the enrichment agent:

```bash
export OPENAI_API_KEY="your-key-here"
python run_agent.py
```

The agent will:
- Search web for model documentation
- Use GPT-4o-mini to analyze and fill empty fields
- Update YAML files with enriched metadata
- Cache results to avoid reprocessing

## What Gets Enriched

For each model with empty fields, the AI adds:
- **Description** - Model capabilities analysis
- **Context/Output Tokens** - Token limits research  
- **Task Type** - Primary use case (code, chat, etc.)
- **Complexity** - Difficulty tier (easy/medium/hard)
- **Function Calling** - API capability detection
- **Model Size** - Parameter count extraction
- **Latency Tier** - Performance assessment

## Key Files

```
model_data/
├── run_agent.py                    # Simple entry point
├── agent/                          # Modular LangGraph agent
│   ├── workflow.py                 # Main workflow orchestration
│   ├── nodes.py                    # Processing nodes (search, AI, validation)
│   ├── tools.py                    # Search & analysis tools
│   ├── state.py                    # LangGraph state models
│   └── utils.py                    # YAML handling & progress tracking
├── config.py                       # Settings & budget limits
├── cost_tracker.py                 # Cost tracking utilities
└── data/                           # Data storage
    ├── provider_models/            # Model YAML files (20 providers)
    └── cache/                      # Processing cache
```

## Cost & Performance

- **~$0.004 per model** using GPT-4o-mini
- **Budget controls** prevent overspending
- **Smart caching** avoids reprocessing
- **89%+ success rate** with web-enhanced research

The enrichment agent uses LangGraph workflow orchestration with intelligent retry logic and quality validation.