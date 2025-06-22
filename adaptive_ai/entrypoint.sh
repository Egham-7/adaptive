#!/bin/sh
set -e

echo "🚀 Starting the application..."

cd /app
exec poetry run python -u adaptive_ai/main.py "$@"
