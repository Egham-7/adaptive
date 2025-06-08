# Stage 1: The Builder
FROM oven/bun:1-alpine AS builder

WORKDIR /app

# Copy package definition files
COPY package.json bun.lock ./

# Install ALL dependencies first. The --production flag can be problematic for Next.js build.
# We explicitly do not use --production here during the initial install for build.
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Set build arguments and environment variables
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

USER bun

# Copy only the essential production artifacts from the "builder" stage.
# Now, after the build, we can safely assume what's needed for runtime is in .next/ and node_modules.
# However, to be absolutely safe with Prisma, keep the explicit copies for it.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/node_modules/.prisma/client ./node_modules/.prisma/client
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["bun", "run", "start"]
