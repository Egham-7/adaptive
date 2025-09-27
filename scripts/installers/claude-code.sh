#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Claude Code Adaptive Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
CLAUDE_PACKAGE="@anthropic-ai/claude-code"
CONFIG_DIR="$HOME/.claude"
API_BASE_URL="https://www.llmadaptive.uk/api"
API_KEY_URL="https://www.llmadaptive.uk/dashboard"
API_TIMEOUT_MS=3000000

# Model override defaults (can be overridden by environment variables)
# Empty strings enable intelligent model routing for optimal cost/performance
DEFAULT_PRIMARY_MODEL=""
DEFAULT_FAST_MODEL=""

# ========================
#       Utility Functions
# ========================

log_info() {
  echo "🔹 $*"
}

log_success() {
  echo "✅ $*"
}

log_error() {
  echo "❌ $*" >&2
}

ensure_dir_exists() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir" || {
      log_error "Failed to create directory: $dir"
      exit 1
    }
  fi
}

create_config_backup() {
  local config_file="$1"

  if [ -f "$config_file" ]; then
    local backup_file="${config_file}.bak"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local timestamped_backup="${config_file}.${timestamp}.bak"

    # Create timestamped backup
    cp "$config_file" "$timestamped_backup" || {
      log_error "Failed to create timestamped backup: $timestamped_backup"
      exit 1
    }

    # Create/update .bak file for easy revert
    cp "$config_file" "$backup_file" || {
      log_error "Failed to create backup: $backup_file"
      exit 1
    }

    log_success "Config backed up to: $backup_file"
    log_info "Timestamped backup: $timestamped_backup"
    log_info "To revert: cp \"$backup_file\" \"$config_file\""
  fi
}

# ========================
#     Node.js Installation Functions
# ========================

install_nodejs() {
  local platform=$(uname -s)

  case "$platform" in
  Linux | Darwin)
    log_info "Installing Node.js on $platform..."

    # Install nvm
    log_info "Installing nvm ($NVM_VERSION)..."
    curl -s https://raw.githubusercontent.com/nvm-sh/nvm/"$NVM_VERSION"/install.sh | bash

    # Load nvm
    log_info "Loading nvm environment..."
    \. "$HOME/.nvm/nvm.sh"

    # Install Node.js
    log_info "Installing Node.js $NODE_INSTALL_VERSION..."
    nvm install "$NODE_INSTALL_VERSION"

    # Verify installation
    node -v &>/dev/null || {
      log_error "Node.js installation failed"
      exit 1
    }
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
    ;;
  *)
    log_error "Unsupported platform: $platform"
    exit 1
    ;;
  esac
}

# ========================
#     Node.js Check Functions
# ========================

check_nodejs() {
  if command -v node &>/dev/null; then
    current_version=$(node -v | sed 's/v//')
    major_version=$(echo "$current_version" | cut -d. -f1)

    if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
      log_success "Node.js is already installed: v$current_version"
      return 0
    else
      log_info "Node.js v$current_version is installed but version < $NODE_MIN_VERSION. Upgrading..."
      install_nodejs
    fi
  else
    log_info "Node.js not found. Installing..."
    install_nodejs
  fi
}

# ========================
#     Claude Code Installation
# ========================

install_claude_code() {
  if command -v claude &>/dev/null; then
    log_success "Claude Code is already installed: $(claude --version)"
  else
    log_info "Installing Claude Code..."
    npm install -g "$CLAUDE_PACKAGE" || {
      log_error "Failed to install claude-code"
      exit 1
    }
    log_success "Claude Code installed successfully"
  fi
}

configure_claude_json() {
  node --eval '
      const os = require("os");
      const fs = require("fs");
      const path = require("path");

      const homeDir = os.homedir();
      const filePath = path.join(homeDir, ".claude.json");
      if (fs.existsSync(filePath)) {
          const content = JSON.parse(fs.readFileSync(filePath, "utf-8"));
          fs.writeFileSync(filePath, JSON.stringify({ ...content, hasCompletedOnboarding: true }, null, 2), "utf-8");
      } else {
          fs.writeFileSync(filePath, JSON.stringify({ hasCompletedOnboarding: true }, null, 2), "utf-8");
      }'
}

# ========================
#     API Key Configuration
# ========================

validate_api_key() {
  local api_key="$1"

  # Basic validation - check if it looks like a valid API key format
  if [[ ! "$api_key" =~ ^[A-Za-z0-9_-]{20,}$ ]]; then
    log_error "API key format appears invalid. Please check your key."
    return 1
  fi
  return 0
}

validate_model_override() {
  local model="$1"

  # Allow empty string for intelligent routing
  if [ -z "$model" ]; then
    return 0
  fi

  # Validate format: provider:model_name
  if [[ ! "$model" =~ ^[a-zA-Z0-9_-]+:[a-zA-Z0-9_.-]+$ ]]; then
    log_error "Model override format invalid. Use format: provider:model_name (e.g., anthropic:claude-sonnet-4-0) or empty string for intelligent routing"
    return 1
  fi
  return 0
}

configure_claude() {
  log_info "Configuring Claude Code for Adaptive..."
  echo "   You can get your API key from: $API_KEY_URL"

  # Check for environment variable first
  local api_key="${ADAPTIVE_API_KEY:-}"

  # Check for model overrides
  local primary_model="${ADAPTIVE_PRIMARY_MODEL:-$DEFAULT_PRIMARY_MODEL}"
  local fast_model="${ADAPTIVE_FAST_MODEL:-$DEFAULT_FAST_MODEL}"

  # Validate model overrides if provided
  if [ "$primary_model" != "$DEFAULT_PRIMARY_MODEL" ]; then
    log_info "Using custom primary model: $primary_model"
    if ! validate_model_override "$primary_model"; then
      log_error "Invalid primary model format in ADAPTIVE_PRIMARY_MODEL"
      exit 1
    fi
  fi

  if [ "$fast_model" != "$DEFAULT_FAST_MODEL" ]; then
    log_info "Using custom fast model: $fast_model"
    if ! validate_model_override "$fast_model"; then
      log_error "Invalid fast model format in ADAPTIVE_FAST_MODEL"
      exit 1
    fi
  fi

  if [ -n "$api_key" ]; then
    log_info "Using API key from ADAPTIVE_API_KEY environment variable"
    if ! validate_api_key "$api_key"; then
      log_error "Invalid API key format in ADAPTIVE_API_KEY environment variable"
      exit 1
    fi
  # Check if running in non-interactive mode (e.g., piped from curl)
  elif [ ! -t 0 ]; then
    echo ""
    log_info "🎯 Interactive setup required for API key configuration"
    echo ""
    echo "📥 Option 1: Download and run interactively (Recommended)"
    echo "   curl -o claude-code.sh https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh"
    echo "   chmod +x claude-code.sh"
    echo "   ./claude-code.sh"
    echo ""
    echo "🔑 Option 2: Set API key via environment variable"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh | bash"
    echo ""
    echo "🎯 Option 3: Customize models (Advanced)"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   export ADAPTIVE_PRIMARY_MODEL='anthropic:claude-opus-4-1'  # or empty for intelligent routing"
    echo "   export ADAPTIVE_FAST_MODEL='anthropic:claude-3-5-haiku-latest'  # or empty for intelligent routing"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/claude-code.sh | bash"
    echo ""
    echo "⚙️  Option 4: Manual configuration (Advanced users)"
    echo "   mkdir -p ~/.claude"
    echo "   cat > ~/.claude/settings.json << 'EOF'"
    echo "{"
    echo '  "env": {'
    echo '    "ANTHROPIC_AUTH_TOKEN": "your_api_key_here",'
    echo '    "ANTHROPIC_BASE_URL": "https://www.llmadaptive.uk/api/v1",'
    echo '    "API_TIMEOUT_MS": "3000000",'
    echo '    "ANTHROPIC_MODEL": "",'
    echo '    "ANTHROPIC_SMALL_FAST_MODEL": ""'
    echo "  }"
    echo "}"
    echo "EOF"
    echo ""
    echo "🔗 Get your API key: $API_KEY_URL"
    exit 1
  else
    # Interactive mode - prompt for API key
    local attempts=0
    local max_attempts=3

    while [ $attempts -lt $max_attempts ]; do
      echo -n "🔑 Please enter your Adaptive API key: "
      read -s api_key
      echo

      if [ -z "$api_key" ]; then
        log_error "API key cannot be empty."
        ((attempts++))
        continue
      fi

      if validate_api_key "$api_key"; then
        break
      fi

      ((attempts++))
      if [ $attempts -lt $max_attempts ]; then
        log_info "Please try again ($((max_attempts - attempts)) attempts remaining)..."
      fi
    done

    if [ $attempts -eq $max_attempts ]; then
      log_error "Maximum attempts reached. Please run the script again."
      exit 1
    fi
  fi

  ensure_dir_exists "$CONFIG_DIR"

  # Create backup of existing configuration
  local settings_file="$CONFIG_DIR/settings.json"
  create_config_backup "$settings_file"

  # Write configuration file
  node --eval '
        const os = require("os");
        const fs = require("fs");
        const path = require("path");

        const homeDir = os.homedir();
        const filePath = path.join(homeDir, ".claude", "settings.json");
        const apiKey = "'"$api_key"'";
        const primaryModel = "'"$primary_model"'";
        const fastModel = "'"$fast_model"'";

        const content = fs.existsSync(filePath)
            ? JSON.parse(fs.readFileSync(filePath, "utf-8"))
            : {};

        fs.writeFileSync(filePath, JSON.stringify({
            ...content,
            env: {
                ANTHROPIC_AUTH_TOKEN: apiKey,
                ANTHROPIC_BASE_URL: "'"$API_BASE_URL"'",
                API_TIMEOUT_MS: "'"$API_TIMEOUT_MS"'",
                ANTHROPIC_MODEL: primaryModel,
                ANTHROPIC_SMALL_FAST_MODEL: fastModel,
            }
        }, null, 2), "utf-8");
    ' || {
    log_error "Failed to write settings.json"
    exit 1
  }

  log_success "Claude Code configured for Adaptive successfully"
  log_info "Configuration saved to: $CONFIG_DIR/settings.json"
}

# ========================
#        Main Flow
# ========================

show_banner() {
  echo "=========================================="
  echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
  echo "=========================================="
  echo "Configure Claude Code to use Adaptive's"
  echo "intelligent LLM routing for 60-80% cost savings"
  echo ""
}

verify_installation() {
  log_info "Verifying installation..."

  # Check if Claude Code can be found
  if ! command -v claude &>/dev/null; then
    log_error "Claude Code installation verification failed"
    return 1
  fi

  # Check if configuration file exists
  if [ ! -f "$CONFIG_DIR/settings.json" ]; then
    log_error "Configuration file not found"
    return 1
  fi

  log_success "Installation verification passed"
  return 0
}

main() {
  show_banner

  check_nodejs
  install_claude_code
  configure_claude_json
  configure_claude

  if verify_installation; then
    echo ""
    echo "╭────────────────────────────────────────────╮"
    echo "│  🎉 Claude Code + Adaptive Setup Complete  │"
    echo "╰────────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start:"
    echo "   claude                    # Start Claude Code with Adaptive routing"
    echo ""
    echo "🔍 Verify Setup:"
    echo "   /status                   # Check Adaptive integration in Claude Code"
    echo "   /help                     # View available commands"
    echo ""
    echo "📊 Monitor Usage:"
    echo "   Dashboard: $API_KEY_URL"
    echo "   API Logs: ~/.claude/logs/"
    echo ""
    echo "💡 Pro Tips:"
    echo "   • Intelligent routing enabled by default for optimal cost/performance"
    echo "   • Current models: Claude Opus 4.1, Sonnet 4, Haiku 3.5"
    echo "   • Override models: ADAPTIVE_PRIMARY_MODEL, ADAPTIVE_FAST_MODEL env vars"
    echo "   • Use provider:model format (e.g. anthropic:claude-opus-4-1)"
    echo ""
    echo "📖 Full Documentation: https://docs.llmadaptive.uk/developer-tools/claude-code"
    echo "🐛 Report Issues: https://github.com/Egham-7/adaptive/issues"
  else
    echo ""
    log_error "❌ Installation verification failed"
    echo ""
    echo "🔧 Manual Setup (if needed):"
    echo "   Configuration: ~/.claude/settings.json"
    echo "   Expected format:"
    echo '   {"env":{"ANTHROPIC_AUTH_TOKEN":"your_key","ANTHROPIC_BASE_URL":"https://www.llmadaptive.uk/api/v1"}}'
    echo ""
    echo "🆘 Get help: https://docs.llmadaptive.uk/troubleshooting"
    exit 1
  fi
}

main "$@"
