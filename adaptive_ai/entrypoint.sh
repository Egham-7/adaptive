#!/bin/sh
set -e

echo "🚀 Starting the application..."

cd /app
exec poetry run python -u main.py "$@"
