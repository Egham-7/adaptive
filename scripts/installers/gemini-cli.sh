#!/bin/bash

set -euo pipefail

# ========================
#       Constants
# ========================
SCRIPT_NAME="Gemini CLI Adaptive Installer"
SCRIPT_VERSION="1.0.0"
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
GEMINI_PACKAGE="@google/gemini-cli"
CONFIG_DIR="$HOME/.gemini"
API_BASE_URL="https://www.llmadaptive.uk/api"
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
#     Runtime Installation Functions
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
    # shellcheck source=/dev/null
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
#     Runtime Check Functions
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

check_runtime() {
  # Check for Node.js and npm
  if command -v node &>/dev/null && command -v npm &>/dev/null; then
    current_version=$(node -v | sed 's/v//')
    major_version=$(echo "$current_version" | cut -d. -f1)

    if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
      log_info "Using Node.js runtime"
      INSTALL_CMD="npm install -g"
      return 0
    fi
  fi

  # Install Node.js if not found or version is too old
  log_info "No suitable runtime found. Installing Node.js..."
  check_nodejs
  INSTALL_CMD="npm install -g"
}

# ========================
#     Gemini CLI Installation
# ========================

install_gemini_cli() {
  if command -v gemini &>/dev/null; then
    log_success "Gemini CLI is already installed: $(gemini --version 2>/dev/null || echo 'installed')"
  else
    log_info "Installing Gemini CLI..."
    $INSTALL_CMD "$GEMINI_PACKAGE" || {
      log_error "Failed to install gemini-cli"
      exit 1
    }
    log_success "Gemini CLI installed successfully"
  fi
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

detect_shell() {
  # Check SHELL environment variable first (most reliable for installer scripts)
  case "${SHELL:-}" in
    */zsh) echo "zsh" ;;
    */bash) echo "bash" ;;
    */fish) echo "fish" ;;
    *)
      # Fallback to checking version variables if SHELL is not set
      if [ -n "${ZSH_VERSION:-}" ]; then
        echo "zsh"
      elif [ -n "${BASH_VERSION:-}" ]; then
        echo "bash"
      elif [ -n "${FISH_VERSION:-}" ]; then
        echo "fish"
      else
        echo "bash" # Default fallback
      fi
      ;;
  esac
}

get_shell_config_file() {
  local shell_type="$1"

  case "$shell_type" in
    zsh)
      echo "$HOME/.zshrc"
      ;;
    bash)
      if [ -f "$HOME/.bashrc" ]; then
        echo "$HOME/.bashrc"
      elif [ -f "$HOME/.bash_profile" ]; then
        echo "$HOME/.bash_profile"
      else
        echo "$HOME/.bashrc"
      fi
      ;;
    fish)
      mkdir -p "$HOME/.config/fish"
      echo "$HOME/.config/fish/config.fish"
      ;;
    *)
      echo "$HOME/.bashrc"
      ;;
  esac
}

add_env_to_shell_config() {
  local api_key="$1"
  local model="$2"
  local base_url="$3"
  local shell_type
  local config_file

  shell_type=$(detect_shell)
  config_file=$(get_shell_config_file "$shell_type")

  log_info "Adding environment variables to $config_file"

  # Create config file if it doesn't exist
  touch "$config_file"

  # Check if GEMINI_API_KEY already exists in the config
  if grep -q "GEMINI_API_KEY" "$config_file" 2>/dev/null; then
    log_info "Gemini environment variables already exist in $config_file, updating..."

    if [ "$shell_type" = "fish" ]; then
      # Fish shell: update API key, base URL, and model
      if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed for Fish
        sed -i '' "s|set -x GEMINI_API_KEY.*|set -x GEMINI_API_KEY \"$api_key\"|" "$config_file"
        sed -i '' "s|set -x GOOGLE_GEMINI_BASE_URL.*|set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"|" "$config_file"
        sed -i '' "s|set -x GEMINI_MODEL.*|set -x GEMINI_MODEL \"$model\"|" "$config_file"
      else
        # Linux sed for Fish
        sed -i "s|set -x GEMINI_API_KEY.*|set -x GEMINI_API_KEY \"$api_key\"|" "$config_file"
        sed -i "s|set -x GOOGLE_GEMINI_BASE_URL.*|set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"|" "$config_file"
        sed -i "s|set -x GEMINI_MODEL.*|set -x GEMINI_MODEL \"$model\"|" "$config_file"
      fi

      # Add GOOGLE_GEMINI_BASE_URL if it doesn't exist in Fish config
      if ! grep -q "GOOGLE_GEMINI_BASE_URL" "$config_file" 2>/dev/null; then
        echo "set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"" >> "$config_file"
      fi
      # Add GEMINI_MODEL if it doesn't exist in Fish config
      if ! grep -q "GEMINI_MODEL" "$config_file" 2>/dev/null; then
        echo "set -x GEMINI_MODEL \"$model\"" >> "$config_file"
      fi
    else
      # POSIX shells (bash/zsh): update API key, base URL, and model
      if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed for bash/zsh
        sed -i '' "s|export GEMINI_API_KEY=.*|export GEMINI_API_KEY=\"$api_key\"|" "$config_file"
        sed -i '' "s|export GOOGLE_GEMINI_BASE_URL=.*|export GOOGLE_GEMINI_BASE_URL=\"$base_url\"|" "$config_file"
        sed -i '' "s|export GEMINI_MODEL=.*|export GEMINI_MODEL=\"$model\"|" "$config_file"
      else
        # Linux sed for bash/zsh
        sed -i "s|export GEMINI_API_KEY=.*|export GEMINI_API_KEY=\"$api_key\"|" "$config_file"
        sed -i "s|export GOOGLE_GEMINI_BASE_URL=.*|export GOOGLE_GEMINI_BASE_URL=\"$base_url\"|" "$config_file"
        sed -i "s|export GEMINI_MODEL=.*|export GEMINI_MODEL=\"$model\"|" "$config_file"
      fi

      # Add GOOGLE_GEMINI_BASE_URL if it doesn't exist in POSIX shell config
      if ! grep -q "GOOGLE_GEMINI_BASE_URL" "$config_file" 2>/dev/null; then
        echo "export GOOGLE_GEMINI_BASE_URL=\"$base_url\"" >> "$config_file"
      fi
      # Add GEMINI_MODEL if it doesn't exist in POSIX shell config
      if ! grep -q "GEMINI_MODEL" "$config_file" 2>/dev/null; then
        echo "export GEMINI_MODEL=\"$model\"" >> "$config_file"
      fi
    fi
  else
    # Add new environment variables based on shell type
    echo "" >> "$config_file"
    echo "# Gemini CLI with Adaptive LLM API Configuration (added by gemini-cli installer)" >> "$config_file"
    if [ "$shell_type" = "fish" ]; then
      echo "set -x GEMINI_API_KEY \"$api_key\"" >> "$config_file"
      echo "set -x GOOGLE_GEMINI_BASE_URL \"$base_url\"" >> "$config_file"
      echo "set -x GEMINI_MODEL \"$model\"" >> "$config_file"
    else
      echo "export GEMINI_API_KEY=\"$api_key\"" >> "$config_file"
      echo "export GOOGLE_GEMINI_BASE_URL=\"$base_url\"" >> "$config_file"
      echo "export GEMINI_MODEL=\"$model\"" >> "$config_file"
    fi
  fi

  log_success "Environment variables added to $config_file"
  if [ -z "$model" ]; then
    log_info "GEMINI_MODEL set to empty for intelligent routing (automatic model selection)"
  else
    log_info "GEMINI_MODEL set to: $model"
  fi
  if [ "$shell_type" = "fish" ]; then
    log_info "Restart your terminal or run 'source $config_file' to apply changes"
  else
    log_info "Run 'source $config_file' or restart your terminal to apply changes"
  fi
}

validate_model_override() {
  local model="$1"

  # Allow empty string for intelligent routing
  if [ -z "$model" ]; then
    return 0
  fi

  # Validate format: provider:model_name
  if [[ ! "$model" =~ ^[a-zA-Z0-9_-]+:[a-zA-Z0-9_.-]+$ ]]; then
    log_error "Model format invalid. Use format: provider:model_name (e.g., gemini:gemini-2.5-pro, gemini:gemini-2.5-flash, anthropic:claude-sonnet-4-20250514) or empty string for intelligent routing"
    return 1
  fi
  return 0
}

configure_gemini() {
  log_info "Configuring Gemini CLI for Adaptive..."
  echo "   You can get your API key from: $API_KEY_URL"

  # Check for environment variable first
  local api_key="${ADAPTIVE_API_KEY:-}"

  # Check for model overrides
  local model="${ADAPTIVE_MODEL:-$DEFAULT_MODEL}"

  # Validate model override if provided
  if [ "$model" != "$DEFAULT_MODEL" ]; then
    log_info "Using custom model: $model"
    if ! validate_model_override "$model"; then
      log_error "Invalid model format in ADAPTIVE_MODEL"
      exit 1
    fi
  fi

  # Use base URL as-is - let Gemini CLI construct the full path
  local base_url="$API_BASE_URL"

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
    echo "   curl -o gemini-cli.sh https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/gemini-cli.sh"
    echo "   chmod +x gemini-cli.sh"
    echo "   ./gemini-cli.sh"
    echo ""
    echo "🔑 Option 2: Set API key via environment variable"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/gemini-cli.sh | bash"
    echo "   # The installer will automatically add the API key to your shell config"
    echo ""
    echo "🎯 Option 3: Customize model (Advanced)"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   export ADAPTIVE_MODEL='gemini:gemini-2.5-flash'  # or empty for intelligent routing"
    echo "   curl -fsSL https://raw.githubusercontent.com/Egham-7/adaptive/main/scripts/installers/gemini-cli.sh | bash"
    echo ""
    echo "⚙️  Option 4: Manual configuration (Advanced users)"
    echo "   mkdir -p ~/.gemini"
    echo "   export ADAPTIVE_API_KEY='your-api-key-here'"
    echo "   # Add to your shell config (~/.bashrc, ~/.zshrc, etc.):"
    echo "   echo 'export GEMINI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
    echo "   echo 'export GOOGLE_GEMINI_BASE_URL=\"https://www.llmadaptive.uk/api\"' >> ~/.bashrc"
    echo "   echo 'export GEMINI_MODEL=\"\"' >> ~/.bashrc  # Empty for intelligent routing"
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

  ensure_dir_exists "$CONFIG_DIR"

  log_success "Gemini CLI configured for Adaptive successfully"
  log_info "Base URL: $base_url"

  # Add environment variables to shell configuration with the constructed base URL
  add_env_to_shell_config "$api_key" "$model" "$base_url"
}

# ========================
#        Main Flow
# ========================

show_banner() {
  echo "=========================================="
  echo "  $SCRIPT_NAME v$SCRIPT_VERSION"
  echo "=========================================="
  echo "Configure Gemini CLI to use Adaptive's"
  echo "intelligent LLM routing for 60-80% cost savings"
  echo ""
}

verify_installation() {
  log_info "Verifying installation..."

  # Check if Gemini CLI can be found
  if ! command -v gemini &>/dev/null; then
    log_error "Gemini CLI installation verification failed"
    return 1
  fi

  log_success "Installation verification passed"
  return 0
}

main() {
  show_banner

  check_runtime
  install_gemini_cli
  configure_gemini

  if verify_installation; then
    echo ""
    echo "╭──────────────────────────────────────────╮"
    echo "│  🎉 Gemini CLI + Adaptive Setup Complete │"
    echo "╰──────────────────────────────────────────╯"
    echo ""
    echo "🚀 Quick Start:"
    echo "   gemini                    # Start Gemini CLI with Adaptive routing"
    echo "   gemini \"help me code\"     # Interactive chat mode"
    echo ""
    echo "🔍 Verify Setup:"
    echo "   gemini --version          # Check Gemini CLI installation"
    echo "   echo \$GEMINI_API_KEY      # Check API key environment variable"
    echo "   echo \$GOOGLE_GEMINI_BASE_URL  # Check base URL configuration"
    echo ""
    echo "💡 Usage Examples:"
    echo "   gemini \"explain this code\""
    echo "   gemini \"create a React component for user authentication\""
    echo "   gemini \"debug my Python script\""
    echo ""
    echo "📊 Monitor Usage:"
    echo "   Dashboard: $API_KEY_URL"
    echo ""
    echo "💡 Pro Tips:"
    echo "   • Your API key is automatically saved to your shell config"
    echo "   • GEMINI_MODEL set to empty for intelligent routing (optimal cost/performance)"
    echo "   • Set GEMINI_MODEL='gemini:gemini-2.5-flash' to override with specific model"
    echo "   • Use provider:model format (e.g., gemini:gemini-2.5-pro, anthropic:claude-sonnet-4-20250514)"
    echo "   • Access to Anthropic Claude, OpenAI, and other providers via Adaptive routing"
    echo ""
    echo "🔄 Load Balancing & Fallbacks:"
    echo "   • Adaptive automatically routes to the best available model"
    echo "   • Higher rate limits through multi-provider load balancing"
    echo "   • Automatic fallbacks if one provider fails"
    echo ""
    echo "📖 Full Documentation: https://docs.llmadaptive.uk/developer-tools/gemini-cli"
    echo "🐛 Report Issues: https://github.com/Egham-7/adaptive/issues"
  else
    echo ""
    log_error "❌ Installation verification failed"
    echo ""
    echo "🔧 Manual Setup (if needed):"
    echo "   Configuration: Set environment variables in your shell config"
    echo "   Expected variables:"
    echo '   export GEMINI_API_KEY="your-adaptive-api-key"'
    echo '   export GOOGLE_GEMINI_BASE_URL="https://www.llmadaptive.uk/api"'
    echo '   export GEMINI_MODEL=""  # Empty for intelligent routing'
    echo ""
    echo "🆘 Get help: https://docs.llmadaptive.uk/troubleshooting"
    exit 1
  fi
}

main "$@"
