#!/bin/sh
set -e

echo "🚀 Starting the application..."

cd /app
exec uv run python -u adaptive_ai/main.py "$@"
