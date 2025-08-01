# Stage 0: build with oven/bun
FROM oven/bun:latest AS builder
WORKDIR /app

# install OpenSSL and CA certificates
RUN apt-get update -y \
  && apt-get install -y openssl ca-certificates 

# install deps & generate Prisma client
COPY package.json bun.lock ./
COPY prisma ./prisma
ENV PRISMA_GENERATE_ONLY_ENGINE=true
RUN bun install --frozen-lockfile

COPY . .
ARG DATABASE_URL
ARG CLERK_PUBLISHABLE_KEY
ARG STRIPE_SECRET_KEY
ENV DATABASE_URL=${DATABASE_URL}
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${CLERK_PUBLISHABLE_KEY}
ENV STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}

# now that openssl is available, generate and build
RUN bunx prisma generate \
  && bun run build

# Stage 1: production runner on Debian slim
FROM oven/bun:latest AS runner

WORKDIR /app

# install runtime deps (openssl + CA bundle)
RUN apt-get update -y \
  && apt-get install -y openssl ca-certificates 

# copy Next.js standalone build & Prisma engines
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/node_modules/@prisma/client \
  ./node_modules/@prisma/client
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
# Copy prisma folder (including migrations and schema)
COPY --from=builder /app/prisma ./prisma

# drop to non-root user
RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs
USER nextjs

EXPOSE 3000
ENV PORT=3000 HOSTNAME="0.0.0.0"
CMD ["sh", "-c", "bunx prisma migrate deploy && bun run start"]
