# Stage 1: The Builder
# Use the official oven/bun image with the 'alpine' tag for a small base.
FROM oven/bun:1-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy package definition files to leverage Docker layer caching
COPY package.json bun.lockb ./

# Install dependencies using the frozen lockfile for reproducible builds
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Consolidate the build steps into a single RUN command.
# This creates a more efficient and atomic build layer.
RUN bunx prisma generate && bun run build

# ---

# Stage 2: The Runner
# Start from a fresh, identical alpine image for the final stage
FROM oven/bun:1-alpine

# Set the working directory
WORKDIR /app

# The oven/bun images come with a non-root 'bun' user.
# Using it is a major security best practice.
USER bun

# Copy only the essential production artifacts from the "builder" stage.
# This is the key to a small and secure final image.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expose the port the app will run on
EXPOSE 3000

# The command to start the production server.
# "next start" is the standard command for a Next.js production build.
CMD ["bun", "run", "start"]
