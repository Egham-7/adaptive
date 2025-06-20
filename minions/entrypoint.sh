#!/bin/bash
set -e

# Execute the litgpt serve command, pointing to the downloaded model
exec litgpt serve "${MODEL_ID}" --port "${LITGPT_SERVE_PORT}" --openai_spec true
