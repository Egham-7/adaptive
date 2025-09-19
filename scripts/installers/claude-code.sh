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
API_BASE_URL="https://www.llmadaptive.uk/api/v1"
API_KEY_URL="https://www.llmadaptive.uk/dashboard"
API_TIMEOUT_MS=3000000

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

configure_claude() {
  log_info "Configuring Claude Code for Adaptive..."
  echo "   You can get your API key from: $API_KEY_URL"

  # Retry loop for API key input
  local attempts=0
  local max_attempts=3
  local api_key=""

  while [ $attempts -lt $max_attempts ]; do
    read -s -p "🔑 Please enter your Adaptive API key: " api_key
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

  ensure_dir_exists "$CONFIG_DIR"

  # Write configuration file
  node --eval '
        const os = require("os");
        const fs = require("fs");
        const path = require("path");

        const homeDir = os.homedir();
        const filePath = path.join(homeDir, ".claude", "settings.json");
        const apiKey = "'"$api_key"'";

        const content = fs.existsSync(filePath)
            ? JSON.parse(fs.readFileSync(filePath, "utf-8"))
            : {};

        fs.writeFileSync(filePath, JSON.stringify({
            ...content,
            env: {
                ANTHROPIC_AUTH_TOKEN: apiKey,
                ANTHROPIC_BASE_URL: "'"$API_BASE_URL"'",
                API_TIMEOUT_MS: "'"$API_TIMEOUT_MS"'",
            }
        }, null, 2), "utf-8");
    ' || {
    log_error "Failed to write settings.json"
    exit 1
  }

  log_success "Claude Code configured for Adaptive successfully"
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
    log_success "🎉 Installation completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Run 'claude' to start Claude Code"
    echo "   2. Type '/status' in Claude Code to verify Adaptive integration"
    echo "   3. Visit $API_KEY_URL to monitor your usage"
    echo ""
    echo "📖 Documentation: https://docs.llmadaptive.uk/developer-tools/claude-code"
  else
    log_error "Installation completed with errors. Please check the configuration manually."
    exit 1
  fi
}

main "$@"

