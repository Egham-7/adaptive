#!/bin/bash

# Azure Container Apps vLLM Deployment Script
# Usage: ./deploy.sh [env-file] [action] [options]
# Example: ./deploy.sh saul-7b.env create
# Example: ./deploy.sh saul-7b.env update --dry-run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENV_FILE=${1:-.env}
ACTION=${2:-create}
DRY_RUN=false
FORCE=false
VERBOSE=true

# Parse additional options
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case $1 in
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --force)
    FORCE=true
    shift
    ;;
  --verbose | -v)
    VERBOSE=true
    shift
    ;;
  --help | -h)
    echo "Usage: $0 [env-file] [action] [options]"
    echo ""
    echo "Arguments:"
    echo "  env-file    Environment file (default: .env)"
    echo "  action      create|update|delete|status (default: create)"
    echo ""
    echo "Options:"
    echo "  --dry-run   Preview the generated YAML without deploying"
    echo "  --force     Skip confirmation prompts"
    echo "  --verbose   Enable verbose output"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 saul-7b.env create"
    echo "  $0 production.env update --dry-run"
    echo "  $0 .env delete --force"
    exit 0
    ;;
  *)
    echo -e "${RED}Error: Unknown option $1${NC}"
    exit 1
    ;;
  esac
done

# Function to print colored output
print_info() {
  echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
  echo -e "${RED}❌ $1${NC}"
}

print_header() {
  echo -e "${BLUE}===========================================${NC}"
  echo -e "${BLUE} Azure Container Apps vLLM Deployment${NC}"
  echo -e "${BLUE}===========================================${NC}"
}

# Function to check prerequisites
check_prerequisites() {
  print_info "Checking prerequisites..."

  # Check if Azure CLI is installed and logged in
  if ! command -v az &>/dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
  fi

  # Check if logged into Azure
  if ! az account show &>/dev/null; then
    print_error "Not logged into Azure. Please run 'az login' first."
    exit 1
  fi

  # Check if envsubst is available
  if ! command -v envsubst &>/dev/null; then
    print_error "envsubst is not installed. Please install gettext package."
    print_info "Ubuntu/Debian: sudo apt-get install gettext-base"
    print_info "macOS: brew install gettext"
    exit 1
  fi

  print_success "Prerequisites check passed"
}

# Function to validate environment file
validate_env_file() {
  if [ ! -f "$ENV_FILE" ]; then
    print_error "Environment file '$ENV_FILE' not found!"
    print_info "Usage: $0 [env-file] [create|update|delete|status]"
    exit 1
  fi

  if [ ! -f "minion.yaml" ]; then
    print_error "minion.yaml not found in current directory!"
    exit 1
  fi

  print_success "Environment file found: $ENV_FILE"
}

# Function to load and validate environment variables
load_environment() {
  print_info "Loading environment variables from: $ENV_FILE"
  source "$ENV_FILE"

  # Validate required variables
  local required_vars=("ENVIRONMENT_ID" "APP_NAME" "HF_TOKEN" "MODEL_ID")
  local missing_vars=()

  for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
      missing_vars+=("$var")
    fi
  done

  if [ ${#missing_vars[@]} -ne 0 ]; then
    print_error "Required environment variables missing:"
    for var in "${missing_vars[@]}"; do
      echo "  - $var"
    done
    exit 1
  fi

  print_success "Environment variables loaded successfully"

  if [ "$VERBOSE" = true ]; then
    echo ""
    print_info "Configuration Summary:"
    echo "  App Name: $APP_NAME"
    echo "  Model: $MODEL_ID"
    echo "  Workload Profile: ${WORKLOAD_PROFILE:-NC24-A100}"
    echo "  Location: ${LOCATION:-swedencentral}"
    echo "  Min Replicas: ${MIN_REPLICAS:-0}"
    echo "  Max Replicas: ${MAX_REPLICAS:-1}"
    echo ""
  fi
}

# Function to generate YAML
generate_yaml() {
  print_info "Generating YAML configuration..."
  TEMP_YAML=$(mktemp /tmp/containerapp-XXXXXX.yaml)

  if ! envsubst <minion.yaml >"$TEMP_YAML"; then
    print_error "Failed to generate YAML configuration"
    rm -f "$TEMP_YAML"
    exit 1
  fi

  print_success "YAML configuration generated: $TEMP_YAML"

  if [ "$VERBOSE" = true ] || [ "$DRY_RUN" = true ]; then
    echo ""
    print_info "Generated YAML preview:"
    echo "========================"
    cat "$TEMP_YAML"
    echo "========================"
    echo ""
  fi
}

# Function to check if container app exists
check_app_exists() {
  local rg=$(echo "$ENVIRONMENT_ID" | cut -d'/' -f5)
  if az containerapp show --name "$APP_NAME" --resource-group "$rg" &>/dev/null; then
    return 0
  else
    return 1
  fi
}

# Function to get resource group from environment ID
get_resource_group() {
  echo "$ENVIRONMENT_ID" | cut -d'/' -f5
}

# Function to get environment name from environment ID
get_environment_name() {
  echo "$ENVIRONMENT_ID" | cut -d'/' -f9
}

# Function to deploy container app
deploy_app() {
  local rg=$(get_resource_group)
  local env_name=$(get_environment_name)

  if [ "$ACTION" = "create" ]; then
    if check_app_exists; then
      print_warning "Container app '$APP_NAME' already exists in resource group '$rg'"
      if [ "$FORCE" = false ]; then
        read -p "Do you want to update it instead? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
          ACTION="update"
        else
          print_info "Deployment cancelled"
          rm -f "$TEMP_YAML"
          exit 0
        fi
      else
        print_info "Switching to update mode due to --force flag"
        ACTION="update"
      fi
    fi
  fi

  if [ "$DRY_RUN" = true ]; then
    print_info "Dry run mode - no actual deployment will be performed"
    print_success "YAML validation passed"
    print_info "Would execute: az containerapp $ACTION --name '$APP_NAME' --resource-group '$rg' --environment '$env_name' --yaml '$TEMP_YAML'"
    rm -f "$TEMP_YAML"
    exit 0
  fi

  if [ "$FORCE" = false ]; then
    echo ""
    print_warning "Ready to deploy Container App with the following configuration:"
    echo "  App Name: $APP_NAME"
    echo "  Action: $ACTION"
    echo "  Resource Group: $rg"
    echo "  Environment: $env_name"
    echo "  Model: $MODEL_ID"
    echo "  Workload Profile: ${WORKLOAD_PROFILE:-NC24-A100}"
    echo ""
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      print_info "Deployment cancelled"
      rm -f "$TEMP_YAML"
      exit 0
    fi
  fi

  print_info "Starting deployment..."

  case $ACTION in
  create)
    if az containerapp create --name "$APP_NAME" --resource-group "$rg" --environment "$env_name" --yaml "$TEMP_YAML"; then
      print_success "Container app created successfully!"
    else
      print_error "Failed to create container app"
      rm -f "$TEMP_YAML"
      exit 1
    fi
    ;;
  update)
    if az containerapp update --name "$APP_NAME" --resource-group "$rg" --yaml "$TEMP_YAML"; then
      print_success "Container app updated successfully!"
    else
      print_error "Failed to update container app"
      rm -f "$TEMP_YAML"
      exit 1
    fi
    ;;
  *)
    print_error "Invalid action: $ACTION"
    rm -f "$TEMP_YAML"
    exit 1
    ;;
  esac
}

# Function to delete container app
delete_app() {
  local rg=$(get_resource_group)

  if ! check_app_exists; then
    print_warning "Container app '$APP_NAME' does not exist in resource group '$rg'"
    exit 0
  fi

  if [ "$FORCE" = false ]; then
    echo ""
    print_warning "This will DELETE the following container app:"
    echo "  App Name: $APP_NAME"
    echo "  Resource Group: $rg"
    echo ""
    read -p "Are you sure you want to delete this app? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      print_info "Deletion cancelled"
      exit 0
    fi
  fi

  print_info "Deleting container app..."
  if az containerapp delete --name "$APP_NAME" --resource-group "$rg" --yes; then
    print_success "Container app deleted successfully!"
  else
    print_error "Failed to delete container app"
    exit 1
  fi
}

# Function to show app status
show_status() {
  local rg=$(get_resource_group)

  if ! check_app_exists; then
    print_warning "Container app '$APP_NAME' does not exist in resource group '$rg'"
    exit 0
  fi

  print_info "Container App Status for: $APP_NAME"
  echo ""

  # Get basic info
  local status=$(az containerapp show --name "$APP_NAME" --resource-group "$rg" --query "properties.provisioningState" -o tsv)
  local fqdn=$(az containerapp show --name "$APP_NAME" --resource-group "$rg" --query "properties.configuration.ingress.fqdn" -o tsv)
  local replicas=$(az containerapp show --name "$APP_NAME" --resource-group "$rg" --query "properties.template.scale" -o json)

  echo "Provisioning State: $status"
  echo "App URL: https://$fqdn"
  echo "Scaling Configuration:"
  echo "$replicas" | jq .

  print_info "Recent logs (last 50 lines):"
  az containerapp logs show --name "$APP_NAME" --resource-group "$rg" --tail 50 || true
}

# Function to cleanup temp files
cleanup() {
  if [ -n "$TEMP_YAML" ] && [ -f "$TEMP_YAML" ]; then
    rm -f "$TEMP_YAML"
  fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
  print_header

  # Validate action
  case $ACTION in
  create | update | delete | status) ;;
  *)
    print_error "Invalid action: $ACTION"
    print_info "Valid actions: create, update, delete, status"
    exit 1
    ;;
  esac

  check_prerequisites

  if [ "$ACTION" != "status" ] && [ "$ACTION" != "delete" ]; then
    validate_env_file
  fi

  load_environment

  case $ACTION in
  create | update)
    generate_yaml
    deploy_app
    ;;
  delete)
    delete_app
    ;;
  status)
    show_status
    ;;
  esac

  if [ "$ACTION" = "create" ] || [ "$ACTION" = "update" ]; then
    echo ""
    print_success "Deployment completed successfully!"

    local rg=$(get_resource_group)
    local fqdn=$(az containerapp show --name "$APP_NAME" --resource-group "$rg" --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null || echo "N/A")

    print_info "App Details:"
    echo "  Name: $APP_NAME"
    echo "  Resource Group: $rg"
    echo "  URL: https://$fqdn"
    echo ""
    print_info "Useful commands:"
    echo "  Check status: $0 $ENV_FILE status"
    echo "  View logs: az containerapp logs show --name $APP_NAME --resource-group $rg --follow"
    echo "  Test health: curl https://$fqdn/health"
  fi
}

# Run main function
main "$@"
