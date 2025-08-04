# Adaptive App

Next.js web application with AI chat interface and API management platform.

## Quick Start

```bash
bun install
cp .env.example .env  # Configure database, auth, and API URLs
bun run db:generate
bun run dev
```

## Features

- **Chat Platform** - Real-time streaming chat with conversation management
- **API Management** - Organization, project, and API key management
- **Analytics** - Usage tracking and cost insights dashboard
- **Authentication** - Clerk-based user authentication
- **Payments** - Stripe subscription integration

## Tech Stack

- **Next.js 15** with App Router and React 19
- **TypeScript** with strict type checking
- **Prisma** ORM with PostgreSQL
- **tRPC** for type-safe APIs
- **Clerk** for authentication
- **Tailwind CSS** for styling
- **Bun** as package manager

## Commands

```bash
bun dev              # Development server
bun build            # Production build
bun check            # Lint and format (Biome)
bun typecheck        # TypeScript checking
bun db:studio        # Database GUI
bun db:generate      # Run migrations
```