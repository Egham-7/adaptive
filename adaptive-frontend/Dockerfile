# Stage 1: Dependency Installation
# Using oven/bun for consistency with Bun environment
FROM oven/bun:latest AS dependencies

WORKDIR /app

# Copy bun lock file and package.json first to leverage Docker cache
COPY package.json bun.lockb ./

# Install dependencies using Bun
# --frozen-lockfile is good practice for CI/CD to ensure reproducible builds
RUN bun install --frozen-lockfile --production

# Stage 2: Application Build
FROM oven/bun:latest AS build

WORKDIR /app

# Copy installed node_modules from the dependencies stage
COPY --from=dependencies /app/node_modules ./node_modules

# Copy the rest of the application source code
COPY . .

# Add build arguments for Next.js public environment variables
# These will be baked into the client-side bundle
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_BASE_API_URL

# Set environment variables for the build process
# Next.js reads these from the build args during 'next build'
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ENV NEXT_PUBLIC_BASE_API_URL=$NEXT_PUBLIC_BASE_API_URL

# Build the Next.js application
RUN bun run build

# Stage 3: Production Image (Runtime)
FROM oven/bun:latest AS runner

# Set environment variables for production runtime
# These will be available on the server-side at runtime
ENV NODE_ENV=production
# Re-define public keys for runtime if accessed on server side,
# or for consistent env setup.
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV NEXT_PUBLIC_BASE_API_URL=${NEXT_PUBLIC_BASE_API_URL}

WORKDIR /app

# Copy the .next directory from the build stage
COPY --from=build /app/.next ./.next

# Copy public assets
COPY --from=build /app/public ./public

# Copy package.json and bun.lockb for `bun run start` to work correctly
# This ensures required production dependencies are listed
COPY package.json bun.lockb ./

# Re-install only production dependencies in the runner stage (optional but good practice)
# Bun handles this efficiently.
RUN bun install --production --frozen-lockfile

# Expose the port Next.js runs on
EXPOSE 3000

# Command to start the Next.js application
CMD ["bun", "run", "start"]
