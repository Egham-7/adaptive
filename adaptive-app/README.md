# Adaptive App

Web application providing AI-powered chat interface and API management platform. Built with Next.js 15 and the T3 stack.

## Quick Start

```bash
# Install dependencies
bun install

# Setup environment
cp .env.example .env
# Add your database URL, Clerk keys, and API endpoints

# Initialize database
bun run db:generate && bun run db:migrate

# Start development
bun run dev
```

Visit [http://localhost:3000](http://localhost:3000)

## Features

- **Chat Platform**: Real-time streaming chat with conversation management
- **API Platform**: Organization and project management with API keys  
- **Authentication**: Secure user authentication with Clerk
- **Subscriptions**: Stripe integration for payment processing

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: Clerk
- **API**: tRPC for type-safe client-server communication
- **Package Manager**: Bun

## Development Commands

```bash
bun run dev          # Start development server
bun run build        # Build for production
bun run typecheck    # Run TypeScript checking
bun run check        # Lint and format code
bun run db:studio    # Open Prisma Studio
```

## Environment Variables

```env
DATABASE_URL="postgresql://..."
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_test_..."
CLERK_SECRET_KEY="sk_test_..."
ADAPTIVE_API_BASE_URL="https://your-backend-url"
STRIPE_SECRET_KEY="sk_test_..."
STRIPE_WEBHOOK_SECRET="whsec_..."
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_..."
```