#!/bin/sh
set -e

echo "🚀 Starting the application..."

exec poetry run python /app/main.py "$@"
