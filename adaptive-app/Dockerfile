# Stage 1: The Builder
FROM oven/bun:1-alpine AS builder

WORKDIR /app

# Copy package definition files
COPY package.json bun.lock ./

# Install ALL dependencies needed for build (including dev dependencies)
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Set build arguments for variables needed during build
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG ADAPTIVE_API_BASE_URL

ENV SKIP_ENV_VALIDATION=true
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV ADAPTIVE_API_BASE_URL=${ADAPTIVE_API_BASE_URL}

# Generate Prisma client and build Next.js app
RUN bunx prisma generate && bun run build

# ---

# Stage 2: The Runner
FROM oven/bun:1-alpine

WORKDIR /app

# Copy essential production artifacts as root first
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/node_modules/.prisma/client ./node_modules/.prisma/client
COPY --from=builder /app/package.json ./package.json

# Copy entrypoint script and make it executable AS ROOT.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER bun

# Re-declare environment variables that are needed at runtime.
# These must be re-declared in this stage to be available.
ARG CLERK_SECRET_KEY
ARG DATABASE_URL

ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV ADAPTIVE_API_BASE_URL=${ADAPTIVE_API_BASE_URL}
ENV CLERK_SECRET_KEY=${CLERK_SECRET_KEY}
ENV DATABASE_URL=${DATABASE_URL}

EXPOSE 3000

# Set the entrypoint to our custom script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# This is now effectively ignored by our hardcoded entrypoint, but it's good for clarity.
CMD ["bun", "run", "start"]
