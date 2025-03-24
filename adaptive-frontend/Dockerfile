# Build stage
FROM oven/bun:latest AS build

# Add the build argument
ARG VITE_CLERK_PUBLISHABLE_KEY
# Make it available as an environment variable during build
ENV VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY

WORKDIR /app

# Copy package files
COPY package.json bun.lockb* ./

# Install dependencies
RUN bun install

# Copy the rest of the application
COPY . .

# Build the application
RUN bun run build

# Format Caddyfile stage
FROM caddy:latest AS caddy-formatter

# Copy the Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

# Format the Caddyfile
RUN caddy fmt --overwrite /etc/caddy/Caddyfile

# Production stage with Caddy
FROM caddy:latest

# Copy formatted Caddyfile from formatter stage
COPY --from=caddy-formatter /etc/caddy/Caddyfile /etc/caddy/Caddyfile

# Copy built assets from the build stage
COPY --from=build /app/dist /app/dist

# Expose port 3000
EXPOSE 3000

# Explicitly start Caddy with the configuration file
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile"]
