#!/bin/sh

# entrypoint.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting entrypoint script..."

# Run Prisma migrations
echo "Running Prisma migrations..."
bunx prisma migrate deploy
echo "Prisma migrations applied successfully."

# Start the Next.js application
echo "Starting Next.js application..."
exec bun run start
