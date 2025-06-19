# Adaptive Frontend

Frontend web application for the Adaptive LLM infrastructure platform. Built with Next.js 15 and the T3 stack.

## Overview

This is the client-side application that provides a web interface for interacting with the Adaptive LLM backend. It includes a chat platform, API key management, and user dashboard functionality.

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: Clerk
- **API**: tRPC for type-safe client-server communication
- **Package Manager**: Bun
- **Deployment**: Vercel

## Prerequisites

- Node.js 18+ or Bun
- PostgreSQL database
- Clerk account for authentication
- Access to Adaptive backend API

## Development Setup

1. **Install dependencies**:
   ```bash
   bun install
   ```

2. **Environment configuration**:
   ```bash
   cp .env.example .env
   ```
   
   Required environment variables:
   ```env
   DATABASE_URL="postgresql://..."
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_test_..."
   CLERK_SECRET_KEY="sk_test_..."
   ADAPTIVE_API_BASE_URL="https://your-backend-url"
   STRIPE_SECRET_KEY="sk_test_..."
   STRIPE_WEBHOOK_SECRET="whsec_..."
   NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_..."
   ```

3. **Database setup**:
   ```bash
   bun run db:generate  # Generate Prisma client
   bun run db:migrate   # Run migrations
   ```

4. **Start development server**:
   ```bash
   bun run dev
   ```

   Application runs at `http://localhost:3000`

## Available Scripts

- `bun run dev` - Start development server with Turbo
- `bun run build` - Build for production
- `bun run start` - Start production server
- `bun run preview` - Build and start production server locally
- `bun run typecheck` - Run TypeScript type checking
- `bun run check` - Run Biome linting/formatting
- `bun run check:write` - Run Biome with auto-fix
- `bun run db:studio` - Open Prisma Studio
- `bun run db:push` - Push schema changes to database

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── _components/        # Page-specific components
│   ├── api/               # API routes
│   ├── chat-platform/     # Chat interface pages
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Landing page
├── components/            # Reusable UI components
├── lib/                   # Utility functions and configurations
├── server/                # Server-side code
│   └── api/               # tRPC routers and procedures
├── hooks/                 # Custom React hooks
├── context/               # React context providers
├── types/                 # TypeScript type definitions
└── styles/                # Global styles
```

## Key Features

### Chat Platform
- Real-time streaming chat interface
- Conversation management (CRUD operations)
- Message persistence with Prisma
- Integration with Adaptive backend via OpenAI-compatible API

### API Key Management
- User API key generation and management
- Integration with backend authentication
- Secure key storage and validation

### Subscription Management
- Stripe-based subscription handling
- Multiple subscription tiers
- Secure payment processing
- Webhook integration for subscription events
- Subscription status tracking and management

### Authentication
- Clerk-based user authentication
- Protected routes and API procedures
- JWT token handling for backend communication

## Database Schema

The application uses PostgreSQL with the following main entities:

- **Conversation**: Chat sessions with user isolation
- **Message**: Individual messages with AI SDK support (annotations, attachments)

Both models include soft deletion (`deletedAt`) and proper indexing for performance.

## API Integration

### tRPC Routers

- `conversationRouter` - Conversation CRUD operations
- `messageRouter` - Message handling and batch operations  
- `apiKeysRouter` - API key management with backend integration

### Backend Communication

The app communicates with the Adaptive backend through:
- `/api/chat` - Streaming chat completions
- API key management endpoints
- Authentication via Clerk JWT tokens

## Deployment

### Vercel

The application is optimized for deployment on Vercel:

1. **Connect repository**: Link your GitHub repository to Vercel
2. **Configure environment variables** in Vercel dashboard:
   - `DATABASE_URL`
   - `ADAPTIVE_API_BASE_URL` 
   - `CLERK_SECRET_KEY`
   - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
3. **Deploy**: Vercel will automatically build and deploy on pushes to main branch

### Build Configuration

The app includes production optimizations:
- Prisma client generation during build
- Database migrations on deployment
- Next.js static optimization where possible

## Development Guidelines

### Code Quality
- TypeScript strict mode enabled
- Biome for linting and formatting
- Type-safe API calls with tRPC
- Prisma for type-safe database operations

### Component Structure
- Use Server Components by default
- Client Components marked with `"use client"`
- Reusable UI components in `/components`
- Page-specific components in `/app/_components`

### Database Operations
- Use Prisma transactions for data consistency
- Implement soft deletion patterns
- Include proper error handling and user authorization

## Contributing

1. Follow TypeScript and React best practices
2. Use the existing component patterns and utilities
3. Run type checking and linting before commits
4. Test authentication flows and API integrations
5. Ensure database migrations are included for schema changes

## Troubleshooting

### Common Issues

- **Database connection errors**: Check `DATABASE_URL` format and accessibility
- **Authentication issues**: Verify Clerk keys and configuration
- **Backend communication**: Ensure `ADAPTIVE_API_BASE_URL` is correct and accessible
- **Build errors**: Run `bun run typecheck` to identify TypeScript issues