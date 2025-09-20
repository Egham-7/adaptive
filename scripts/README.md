# Adaptive Installation Scripts

This directory contains installation and configuration scripts for integrating various developer tools with Adaptive's intelligent LLM routing.

## Available Scripts

### Developer Tools

| Tool | Script | Description |
|------|--------|-------------|
| Claude Code | `installers/claude-code.sh` | Configure Claude Code to use Adaptive API |
| OpenAI Codex | `installers/codex.sh` | Configure OpenAI Codex to use Adaptive routing |
| Grok CLI | `installers/grok-cli.sh` | Configure Grok CLI to use Adaptive routing |
| OpenCode | `installers/opencode.sh` | Configure OpenCode to use Adaptive integration |

### Usage

#### Claude Code Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh
chmod +x claude-code.sh
./claude-code.sh
```

#### OpenAI Codex Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/codex.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/codex.sh
chmod +x codex.sh
./codex.sh
```

#### Grok CLI Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/grok-cli.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/grok-cli.sh
chmod +x grok-cli.sh
./grok-cli.sh
```

#### OpenCode Setup
```bash
# Download and run
curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh | bash

# Or download first
curl -O https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh
chmod +x opencode.sh
./opencode.sh
```

## Script Structure

Each installer script follows this pattern:
- **Prerequisites check**: Verify required dependencies (Node.js, etc.)
- **Tool installation**: Install the developer tool if not present
- **Configuration**: Set up Adaptive API integration
- **Verification**: Test the connection and configuration

## Adding New Tools

To add support for a new developer tool:

1. Create `installers/{tool-name}.sh`
2. Follow the existing script structure
3. Update this README with the new tool
4. Add documentation in `adaptive-docs/developer-tools/{tool-name}.mdx`

## Common Configuration

All scripts configure tools to use:
- **API Base URL**: `https://www.llmadaptive.uk/api/v1`
- **Authentication**: User's Adaptive API key
- **Timeout**: 3000000ms for long-running requests

## Support

For issues with installation scripts:
- Check the tool-specific documentation in `adaptive-docs/developer-tools/`
- Visit [docs.llmadaptive.uk](https://docs.llmadaptive.uk)
- Contact support at [support@llmadaptive.uk](mailto:support@llmadaptive.uk)