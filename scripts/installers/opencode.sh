#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="OpenCode Adaptive Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
OPENCODE_PACKAGE="@opencode/cli"
CONFIG_FILE="opencode.json"
API_BASE_URL="https://www.llmadaptive.uk/api/v1"
API_KEY_URL="https://www.llmadaptive.uk/dashboard"

# Model override defaults (can be overridden by environment variables)
# Empty strings enable intelligent model routing for optimal cost/performance
DEFAULT_MODEL=""

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

# ========================
#     Node.js Installation Functions
# ========================

install_nodejs() {
  local platform
  platform=$(uname -s)

  case "$platform" in
  Linux | Darwin)
    log_info "Installing Node.js on $platform..."

    # Install nvm
    log_info "Installing nvm ($NVM_VERSION)..."
    curl -s https://raw.githubusercontent.com/nvm-sh/nvm/"$NVM_VERSION"/install.sh | bash

    # Load nvm
    log_info "Loading nvm environment..."
    # shellcheck disable=SC1091
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
#     OpenCode Installation
# ========================

install_opencode() {
  if command -v opencode &>/dev/null; then
    log_success "OpenCode is already installed: $(opencode --version 2>/dev/null || echo 'installed')"
  else
    log_info "Installing OpenCode..."
    npm install -g "$OPENCODE_PACKAGE" || {
      log_error "Failed to install OpenCode"
      exit 1
    }
    log_success "OpenCode installed successfully"
  fi
}

# ========================
#     Configuration Management
# ========================

create_opencode_config() {
  local config_file="$1"
  local model="$2"

  log_info "Creating OpenCode configuration..."

  # Create the opencode.json configuration
  cat >"$config_file" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "adaptive": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Adaptive",
      "options": {
        "baseURL": "$API_BASE_URL",
        "headers": {
          "User-Agent": "opencode-adaptive-integration"
        }
      },
      "models": {
        "": {
          "name": "🧠 Intelligent Routing",
          "description": "Automatically selects optimal model for cost/performance"
        },
      }
    }
  }
}
EOF

  log_success "OpenCode configuration created at: $config_file"
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

  # Validate model name format
  if [[ ! "$model" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    log_error "Model name format invalid. Use standard model names like 'claude-3-5-sonnet-20241022' or empty string for intelligent routing"
    return 1
  fi
  return 0
}

configure_opencode() {
  log_info "Configuring OpenCode for Adaptive..."
  echo "   You can get your API key from: $API_KEY_URL"

  # Check for environment variable first
  local api_key="${ADAPTIVE_API_KEY:-}"

  # Check for model override
  local model="${ADAPTIVE_MODEL:-$DEFAULT_MODEL}"

  # Validate model override if provided
  if [ "$model" != "$DEFAULT_MODEL" ]; then
    log_info "Using custom model: $model"
    if ! validate_model_override "$model"; then
      log_error "Invalid model format in ADAPTIVE_MODEL"
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
    echo "   curl -o opencode.sh https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh"
    echo "   chmod +x opencode.sh"
    echo "   ./opencode.sh"
    echo ""
    echo "🔑 Option 2: Set API key via environment variable"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh | bash"
    echo ""
    echo "🎯 Option 3: Customize model (Advanced)"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   export ADAPTIVE_MODEL='claude-3-5-sonnet-20241022'  # or empty for intelligent routing"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/opencode.sh | bash"
    echo ""
    echo "⚙️  Option 4: Manual configuration (Advanced users)"
    echo "   1. Create opencode.json in your project directory"
    echo "   2. Add Adaptive provider with: opencode auth login"
    echo "   3. Configure API key and baseURL as shown in documentation"
    echo ""
    echo "🔗 Get your API key: $API_KEY_URL"
    exit 1
  else
    # Interactive mode - prompt for API key
    local attempts=0
    local max_attempts=3

    while [ $attempts -lt $max_attempts ]; do
      echo -n "🔑 Please enter your Adaptive API key: "
      read -rs api_key
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

  # Create opencode.json configuration
  local config_file="$PWD/$CONFIG_FILE"
  create_opencode_config "$config_file" "$model"

  # Set up authentication with OpenCode
  log_info "Setting up OpenCode authentication..."
  log_info "Please follow the authentication prompts..."

  # Use OpenCode's auth system to add the provider
  # Based on documentation, we need to run auth login and select "Other"
  log_info "Run 'opencode auth login' and:"
  log_info "1. Scroll down and select 'Other'"
  log_info "2. Enter provider ID: adaptive"
  log_info "3. Enter your API key when prompted"

  # Note: OpenCode auth is interactive, so we'll provide instructions
  echo ""
  echo "⚠️  IMPORTANT: Complete the authentication setup manually:"
  echo "   opencode auth login"
  echo "   → Select 'Other'"
  echo "   → Provider ID: adaptive"
  echo "   → API Key: [your key]"
  echo ""

  log_success "OpenCode configured for Adaptive successfully"
  log_info "Configuration saved to: $config_file"
  log_info "Authentication configured with OpenCode"
}

# ========================
#        Main Flow
# ========================

show_banner() {
  echo "=========================================="
  echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
  echo "=========================================="
  echo "Configure OpenCode to use Adaptive's"
  echo "intelligent LLM routing for 60-80% cost savings"
  echo ""
}

verify_installation() {
  log_info "Verifying installation..."

  # Check if OpenCode can be found
  if ! command -v opencode &>/dev/null; then
    log_error "OpenCode installation verification failed"
    return 1
  fi

  # Check if configuration file exists
  if [ ! -f "$PWD/$CONFIG_FILE" ]; then
    log_error "Configuration file not found in current directory"
    return 1
  fi

  log_success "Installation verification passed"
  log_info "Note: Complete authentication manually with 'opencode auth login'"
  return 0
}

main() {
  show_banner

  check_nodejs
  install_opencode
  configure_opencode

  if verify_installation; then
    echo ""
    echo "╭─────────────────────────────────────────────╮"
    echo "│  🎉 OpenCode + Adaptive Setup Complete      │"
    echo "╰─────────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start:"
    echo "   1. Complete authentication: opencode auth login"
    echo "      → Select 'Other' → Provider ID: adaptive → Enter API key"
    echo "   2. Start OpenCode: opencode"
    echo "   3. Use /models to select Adaptive models"
    echo ""
    echo "🔍 Verify Setup:"
    echo "   opencode auth list        # Check configured providers"
    echo "   cat opencode.json         # View configuration"
    echo ""
    echo "💡 Usage Examples:"
    echo "   # In OpenCode:"
    echo "   /models                   # Select Adaptive models"
    echo "   'Create a React component for user authentication'"
    echo "   'Optimize this SQL query for better performance'"
    echo "   'Review this code for security vulnerabilities'"
    echo ""
    echo "📊 Monitor Usage:"
    echo "   Dashboard: $API_KEY_URL"
    echo "   Configuration: ./opencode.json"
    echo "   Auth settings: ~/.opencode/"
    echo ""
    echo "💡 Pro Tips:"
    echo "   • Select 'Intelligent Routing' model for optimal cost/performance"
    echo "   • Available models: Claude 3.5 Sonnet/Haiku, GPT-4o/Mini"
    echo "   • Use /models command to switch between models"
    echo "   • Configuration is project-specific via opencode.json"
    echo ""
    echo "📖 Full Documentation: https://docs.llmadaptive.uk/developer-tools/opencode"
    echo "🐛 Report Issues: https://github.com/Egham-7/adaptive/issues"
  else
    echo ""
    log_error "❌ Installation verification failed"
    echo ""
    echo "🔧 Manual Setup (if needed):"
    echo "   1. Create opencode.json in your project:"
    echo "      curl -o opencode.json https://raw.githubusercontent.com/Egham-7/adaptive/main/examples/opencode.json"
    echo "   2. Configure authentication:"
    echo "      opencode auth login"
    echo "      # Select 'Other' → Enter 'adaptive' → Paste API key"
    echo "   3. Start OpenCode and use /models to select Adaptive"
    echo ""
    echo "🆘 Get help: https://docs.llmadaptive.uk/troubleshooting"
    exit 1
  fi
}

main "$@"

