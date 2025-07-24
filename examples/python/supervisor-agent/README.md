# Supervisor Agent

A multi-agent AI system built with LangGraph that coordinates specialized agents for different tasks. The supervisor intelligently routes requests to the most appropriate agent and coordinates complex workflows.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different domains
- **Intelligent Routing**: Supervisor analyzes requests and routes to appropriate agents
- **CLI Interface**: Multiple interaction modes (direct commands, interactive chat)
- **LangGraph Integration**: Built on LangGraph for robust agent coordination
- **Extensible Design**: Easy to add new agents and capabilities

## 🤖 Available Agents

### Code Agent
- **Code Generation**: Generate code in various programming languages
- **Debugging**: Debug code and suggest fixes
- **Testing**: Create unit tests for functions and classes
- **Code Analysis**: Explain and analyze code structure
- **Syntax Validation**: Check code syntax and report errors

### Data Agent  
- **Data Analysis**: Analyze datasets and provide statistical insights
- **Visualizations**: Create charts, graphs, and plots (with automatic temp file cleanup)
- **Statistical Calculations**: Calculate mean, median, standard deviation, etc.
- **CSV Processing**: Handle CSV data operations
- **Mathematical Calculations**: Perform complex mathematical operations
- **Resource Management**: Automatic cleanup of temporary visualization files

### File Agent
- **File Operations**: Read, write, create, delete files
- **Directory Management**: List, create, manage directories
- **File Search**: Search for text patterns within files
- **Content Analysis**: AI-powered analysis of file contents
- **System Operations**: File copying, moving, information retrieval

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd examples/python/supervisor-agent
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   uv install
   ```
   
   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   
   Or create a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key" > .env
   ```

4. **Verify installation:**
   ```bash
   supervisor-agent --config-check
   ```

## 🎯 Usage

### Command Line Interface

#### Direct Commands
Execute single requests directly:

```bash
# Code generation
supervisor-agent "Generate a Python function to sort a list of dictionaries"

# Data analysis
supervisor-agent "Calculate statistics for these numbers: [1, 5, 10, 15, 20]"

# File operations
supervisor-agent "List all Python files in the current directory"

# Multi-agent tasks
supervisor-agent "Create a Python script to analyze CSV data and generate a report"
```

#### Interactive Mode
Start an interactive chat session:

```bash
supervisor-agent --interactive
# or
supervisor-agent -i
```

In interactive mode, you can:
- Have natural conversations with the agents
- Get help with `help` command
- Check system status with `status` command
- Exit with `exit`, `quit`, or `bye`

#### System Status
Check the health and status of all agents:

```bash
supervisor-agent --status
```

### Python API

Use the supervisor programmatically:

```python
from supervisor_agent import SupervisorAgent

# Initialize the supervisor (uses SimpleSupervisorAgent by default)
supervisor = SupervisorAgent()

# Process a request
response = supervisor.process_request("Generate a Python function to calculate fibonacci numbers")
print(response)

# Check agent capabilities
capabilities = supervisor.get_agent_capabilities()
print(capabilities)

# Health check
health = supervisor.health_check()
print(health)

# Resource management - cleanup temporary files created by data agent
from supervisor_agent.agents.data_agent import DataAgent
data_agent = DataAgent()
temp_file_count = data_agent.get_temp_file_count()
print(f"Temporary files: {temp_file_count}")
deleted_files = data_agent.cleanup_temp_files()
print(f"Cleaned up {deleted_files} temporary files")

# Alternative: Use the full LangGraph implementation
from supervisor_agent import LangGraphSupervisorAgent
advanced_supervisor = LangGraphSupervisorAgent()
```

## 📝 Examples

### Code Agent Examples

```bash
# Generate code
supervisor-agent "Write a Python class for a basic calculator"

# Debug code  
supervisor-agent "Debug this code: def add(a, b): return a + c"

# Create tests
supervisor-agent "Generate unit tests for a bubble sort function"

# Explain code
supervisor-agent "Explain what this code does: lambda x: x**2 + 2*x + 1"
```

### Data Agent Examples

```bash
# Statistical analysis
supervisor-agent "Analyze this dataset: [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]"

# Create visualizations
supervisor-agent "Create a bar chart for monthly sales: Jan=100, Feb=120, Mar=90"

# Mathematical calculations
supervisor-agent "Calculate compound interest: principal=1000, rate=5%, time=3 years"

# Data processing
supervisor-agent "Process this CSV data and find outliers: name,age,salary Alice,25,50000 Bob,30,60000"
```

### File Agent Examples

```bash
# File operations
supervisor-agent "Read the contents of config.json"
supervisor-agent "Create a backup directory and copy all .py files there"

# Directory management
supervisor-agent "List all files in /home/user/documents with .txt extension"

# File analysis
supervisor-agent "Analyze the content of README.md and summarize it"

# Search operations
supervisor-agent "Search for the word 'function' in all Python files in src/"
```

### Multi-Agent Workflows

```bash
# Complex data pipeline
supervisor-agent "Read data from sales.csv, calculate monthly trends, create visualizations, and save the analysis to a report"

# Code development workflow
supervisor-agent "Generate a Python data processing script, create tests for it, and save everything to appropriate files"

# File analysis and reporting
supervisor-agent "Analyze all log files in the logs/ directory, identify error patterns, and create a summary report"
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: "gpt-4o-mini")
- `TEMPERATURE`: Response temperature (default: 0.1)
- `MAX_TOKENS`: Maximum tokens per response (optional)
- `VERBOSE`: Enable verbose logging (default: false)

### Configuration File

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.1
VERBOSE=false
```

## 🧪 Running Tests

Run the test suite:

```bash
# Using uv
uv run pytest

# Using pytest directly
pytest

# With coverage
pytest --cov=supervisor_agent
```

## 🎮 Demo

Run the interactive demo to see all capabilities:

```bash
python -m supervisor_agent.examples.demo
```

The demo includes:
- Code agent examples
- Data agent examples  
- File agent examples
- Multi-agent coordination examples
- Request routing analysis

## 🛠️ Development

### Project Structure

```
supervisor-agent/
├── supervisor_agent/
│   ├── __init__.py
│   ├── main.py                 # CLI interface
│   ├── agents/                 # Individual agent implementations
│   │   ├── code_agent.py
│   │   ├── data_agent.py
│   │   └── file_agent.py
│   ├── supervisor/             # Supervisor logic
│   │   ├── supervisor.py
│   │   └── tools.py
│   ├── utils/                  # Utilities and configuration
│   │   └── config.py
│   └── examples/               # Usage examples
│       └── demo.py
├── tests/                      # Test files
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

### Adding New Agents

1. Create a new agent class in `supervisor_agent/agents/`
2. Implement the required methods: `__init__`, `get_system_message`, `can_handle`
3. Add tools using the `@tool` decorator
4. Update the supervisor to include the new agent
5. Add handoff tools for the new agent

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting  
- **MyPy**: Type checking
- **Pytest**: Testing

Run quality checks:
```bash
uv run black supervisor_agent/
uv run ruff check supervisor_agent/
uv run mypy supervisor_agent/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite and quality checks
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Set the `OPENAI_API_KEY` environment variable
- Or create a `.env` file with your API key

**"Module not found" errors**
- Install dependencies: `uv install` or `pip install -e .`
- Ensure you're in the correct directory

**Agent initialization failures**
- Check your OpenAI API key is valid
- Verify internet connectivity
- Check API usage limits

**File permission errors**
- Ensure the application has read/write permissions
- Check file paths are correct and accessible

### Getting Help

1. Run `supervisor-agent --status` to check system health
2. Use `supervisor-agent --config-check` to verify configuration
3. Try the interactive mode: `supervisor-agent -i`
4. Run the demo: `python -m supervisor_agent.examples.demo`

### Debug Mode

Enable verbose output:
```bash
export VERBOSE=true
supervisor-agent -v "your request here"
```

## 🏗️ Architecture

The system uses a hub-and-spoke architecture with the supervisor at the center:

```
    User Request
         ↓
    Supervisor
    /    |    \
Code    Data   File
Agent   Agent  Agent
```

### Flow:
1. User submits request via CLI or API
2. Supervisor analyzes the request
3. Supervisor routes to appropriate agent(s)
4. Agent(s) process the task using their tools
5. Results are returned to supervisor
6. Supervisor aggregates and returns final response

This architecture allows for:
- **Scalability**: Easy to add new agents
- **Modularity**: Each agent is independent
- **Flexibility**: Complex workflows can involve multiple agents
- **Maintainability**: Clear separation of concerns