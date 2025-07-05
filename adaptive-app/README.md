# Adaptive

AI-powered platform with chat interface, API management, and organization tools. Built with Next.js 15 and the T3 stack.

## Quick Start

1. **Install dependencies**:
   ```bash
   bun install
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Add your database URL, Clerk keys, and API endpoints
   ```

3. **Initialize database**:
   ```bash
   bun run db:generate && bun run db:migrate
   ```

4. **Start development**:
   ```bash
   bun run dev
   ```

   Open [http://localhost:3000](http://localhost:3000)

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: Clerk
- **API**: tRPC for type-safe client-server communication
- **Package Manager**: Bun

## Features

- **Chat Platform**: Real-time streaming chat interface with conversation management
- **API Platform**: Organization and project management with API keys
- **Authentication**: Secure user authentication with Clerk
- **Subscription Management**: Stripe integration for payment processing

## Development

```bash
# Common commands
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.