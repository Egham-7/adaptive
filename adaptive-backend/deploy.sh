#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Configuration
ACR_NAME="adaptiveregistry"
CONTAINER_APP_NAME="backend-go"
RESOURCE_GROUP="adaptive"
IMAGE_TAG=$(git rev-parse --short HEAD) # Use git commit hash as tag
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="deployment_${TIMESTAMP}.log"

# Function for logging
log() {
  local message="$1"
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
log "Checking prerequisites..."
for cmd in docker az git; do
  if ! command_exists $cmd; then
    log "ERROR: $cmd is not installed. Please install it and try again."
    exit 1
  fi
done

# Verify Azure login status
if ! az account show &>/dev/null; then
  log "ERROR: Not logged into Azure. Please run 'az login' first."
  exit 1
fi

# Verify ACR exists
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  log "ERROR: ACR '$ACR_NAME' not found in resource group '$RESOURCE_GROUP'."
  exit 1
fi

# Verify Container App exists
if ! az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  log "ERROR: Container App '$CONTAINER_APP_NAME' not found in resource group '$RESOURCE_GROUP'."
  exit 1
fi

# Verify required environment variables
for var in GROQ_API_KEY OPENAI_API_KEY DEEPSEEK_API_KEY CLERK_SECRET_KEY BACKEND_PYTHON_FQDN FRONTEND_FQDN; do
  if [ -z "${!var}" ]; then
    log "ERROR: Environment variable $var is not set."
    exit 1
  fi
done

# Login to ACR
log "Logging into Azure Container Registry..."
if ! az acr login --name "$ACR_NAME"; then
  log "ERROR: Failed to log into ACR '$ACR_NAME'."
  exit 1
fi

# Build and push Docker image
log "Building Docker image for linux/amd64 platform..."
if ! docker build --platform linux/amd64 -t "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG" .; then
  log "ERROR: Docker build failed."
  exit 1
fi

log "Pushing Docker image to ACR..."
if ! docker push "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG"; then
  log "ERROR: Failed to push image to ACR."
  exit 1
fi

# Ensure ACR credentials are set for Container App
log "Ensuring Container App has ACR credentials..."
ACR_USERNAME=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

if [ -z "$ACR_USERNAME" ] || [ -z "$ACR_PASSWORD" ]; then
  log "ERROR: Failed to retrieve ACR credentials."
  exit 1
fi

if ! az containerapp registry set \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --server "$ACR_NAME.azurecr.io" \
  --username "$ACR_USERNAME" \
  --password "$ACR_PASSWORD"; then
  log "WARNING: Failed to set registry credentials. Continuing anyway as they might already be set."
fi

# Deploy to Container App
log "Deploying to Container App..."
if ! az containerapp update \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG" \
  --set-env-vars \
  "GROQ_API_KEY=$GROQ_API_KEY" \
  "OPENAI_API_KEY=$OPENAI_API_KEY" \
  "DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY" \
  "CLERK_SECRET_KEY=$CLERK_SECRET_KEY" \
  "ADDR=:8080" \
  "ENV=production" \
  "ADAPTIVE_AI_BASE_URL=$BACKEND_PYTHON_FQDN" \
  "ALLOWED_ORIGINS=$FRONTEND_FQDN"; then
  log "ERROR: Failed to update Container App."
  exit 1
fi

# Verify deployment
log "Verifying deployment..."
DEPLOYMENT_STATUS=$(az containerapp revision list \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].properties.status" -o tsv)

if [ "$DEPLOYMENT_STATUS" == "Running" ]; then
  log "✅ Deployment successful! Container App is running."
else
  log "⚠️ Deployment completed, but Container App status is: $DEPLOYMENT_STATUS"
  log "Check the Azure portal for more details."
fi

log "Deployment complete! Image: $ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG"
