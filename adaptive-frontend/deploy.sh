#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Configuration
ACR_NAME="adaptiveregistry"
CONTAINER_APP_NAME="frontend"
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

# Check prerequisites
log "Checking prerequisites..."
for cmd in docker az git; do
  if ! command -v $cmd >/dev/null 2>&1; then
    log "ERROR: $cmd is not installed. Please install it and try again."
    exit 1
  fi
done

# Set up Docker Buildx for multi-platform builds
log "Setting up Docker Buildx..."
docker buildx inspect --bootstrap
#
# Login to Azure Container Registry
log "Logging into Azure Container Registry..."
if ! az acr login --name "$ACR_NAME"; then
  log "ERROR: Failed to log into ACR '$ACR_NAME'."
  exit 1
fi

log "Building and pushing Docker image..."
docker buildx build \
  --platform linux/amd64 \
  --tag "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG" \
  --build-arg VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY \
  --push \
  --file Dockerfile \
  .

if [ $? -ne 0 ]; then
  log "ERROR: Failed to build and push image."
  exit 1
fi

log "Image built and pushed successfully: $ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG"

# Ensure ACR credentials are set for Container App
log "Ensuring Container App has ACR credentials..."
ACR_USERNAME=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

if [ -z "$ACR_USERNAME" ] || [ -z "$ACR_PASSWORD" ]; then
  log "ERROR: Failed to retrieve ACR credentials."
  exit 1
fi

az containerapp registry set \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --server "$ACR_NAME.azurecr.io" \
  --username "$ACR_USERNAME" \
  --password "$ACR_PASSWORD" || log "WARNING: Failed to set registry credentials. Continuing anyway as they might already be set."

# Deploy to Container App
log "Deploying to Container App..."
if ! az containerapp update \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG" \
  --set-env-vars \
  "VITE_BASE_API_URL"="$BACKEND_GO_FQDN" \
  "VITE_CLERK_PUBLISHABLE_KEY"="$VITE_CLERK_PUBLISHABLE_KEY" \
  "PORT=3000"; then
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
