FROM node:alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

# Install dependencies needed for building native modules
RUN apk add --no-cache libc6-compat openssl

# Install pnpm globally
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy package manager files
COPY package.json pnpm-lock.yaml ./
COPY prisma ./prisma

# Install dependencies
RUN pnpm install --frozen-lockfile


# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Install pnpm in builder
RUN corepack enable && corepack prepare pnpm@latest --activate

# Set build-time environment variables
ARG DATABASE_URL
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
ARG STRIPE_SECRET_KEY
ARG REDIS_URL
ARG RESEND_API_KEY
ARG JWT_SECRET
ARG ADAPTIVE_API_BASE_URL
ARG NODE_ENV

ENV DATABASE_URL=${DATABASE_URL}
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY}
ENV STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
ENV REDIS_URL=${REDIS_URL}
ENV RESEND_API_KEY=${RESEND_API_KEY}
ENV JWT_SECRET=${JWT_SECRET}
ENV ADAPTIVE_API_BASE_URL=${ADAPTIVE_API_BASE_URL}
ENV NODE_ENV=${NODE_ENV}}

# Generate Prisma client and build Next.js app
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN pnpm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache openssl curl && \
  corepack enable && corepack prepare pnpm@latest --activate

# Create nextjs user and group
RUN addgroup --system --gid 1001 nodejs && \
  adduser --system --uid 1001 nextjs

# Set correct permissions for precompiled standalone
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Copy Prisma files with correct ownership
COPY --from=builder --chown=nextjs:nodejs /app/prisma ./prisma
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

# Switch to nextjs user
USER nextjs


# Expose port
EXPOSE 3000

# Set runtime environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

# Start the application
CMD ["node", "server.js"]
