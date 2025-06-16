#!/bin/bash
set -e

# Base directory for storing downloaded models
DOWNLOAD_BASE_DIR="/app/downloaded_models"
MODEL_LOCAL_PATH="${DOWNLOAD_BASE_DIR}/${MODEL_ID}"

echo "Starting LitGPT server with MODEL_ID: ${MODEL_ID}"

if [ ! -d "$MODEL_LOCAL_PATH" ]; then
  echo "Model not found locally. Downloading ${MODEL_ID} to ${DOWNLOAD_BASE_DIR}..."
  mkdir -p "${DOWNLOAD_BASE_DIR}"

  litgpt download "${MODEL_ID}" --checkpoint_dir "${DOWNLOAD_BASE_DIR}"
  echo "Download complete."
else
  echo "Model already exists at ${MODEL_LOCAL_PATH}. Skipping download."
fi

# Execute the litgpt serve command, pointing to the downloaded model
exec litgpt serve "${MODEL_ID}" --port "${LITGPT_SERVE_PORT}" --openai_spec true
