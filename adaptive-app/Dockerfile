# Stage 0: build with Node.js and pnpm
FROM node:latest AS builder
WORKDIR /app
# install OpenSSL and CA certificates
RUN apt-get update -y \
  && apt-get install -y openssl ca-certificates 
# install pnpm and deps & generate Prisma client
RUN npm install -g pnpm
COPY package.json pnpm-lock.yaml ./
COPY prisma ./prisma
ENV PRISMA_GENERATE_ONLY_ENGINE=true
RUN pnpm install --frozen-lockfile
COPY . .

# Add ALL the build-time environment variables
ARG DATABASE_URL
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
ARG STRIPE_SECRET_KEY
ARG REDIS_URL
ARG RESEND_API_KEY
ARG JWT_SECRET
ARG ADAPTIVE_API_BASE_URL

# Set them as environment variables for the build
ENV DATABASE_URL=${DATABASE_URL}
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY}
ENV STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
ENV REDIS_URL=${REDIS_URL}
ENV RESEND_API_KEY=${RESEND_API_KEY}
ENV JWT_SECRET=${JWT_SECRET}
ENV ADAPTIVE_API_BASE_URL=${ADAPTIVE_API_BASE_URL}

# now that openssl is available, generate and build
RUN pnpm run build

# Stage 1: production runner on Node.js
FROM node:latest AS runner
WORKDIR /app
# install runtime deps (openssl + CA bundle)
RUN apt-get update -y \
  && apt-get install -y openssl ca-certificates 
# Install pnpm in runner stage (before switching to non-root user)
RUN npm install -g pnpm
# copy Next.js standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
# Copy prisma folder (including migrations, schema, and custom generated client)
COPY --from=builder /app/prisma ./prisma
# drop to non-root user
RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs
USER nextjs
EXPOSE 3000
ENV PORT=3000 HOSTNAME="0.0.0.0"
CMD ["pnpm", "run", "start"]
