# Adaptive App - Next.js Frontend

## Overview

The adaptive-app is a modern Next.js frontend that provides a comprehensive web interface for the Adaptive LLM infrastructure. Built with React 19, Next.js 15, and TypeScript, it offers multi-tenant chat capabilities, real-time analytics dashboards, credit management, and API key administration.

## Key Features

- **Multi-Tenant Architecture**: Organization → Project → API Key hierarchy
- **Real-Time Chat Interface**: Streaming chat with AI SDK integration
- **Analytics Dashboards**: Usage tracking, cost analysis, and performance metrics
- **Credit System**: Stripe-powered billing and credit management
- **API Key Management**: Secure API key generation and administration
- **Provider Comparison**: Real-time cost and performance comparisons
- **Responsive Design**: Mobile-first design with dark/light mode support

## Technology Stack

- **Framework**: Next.js 15 with App Router and React 19
- **Database**: PostgreSQL with Prisma ORM and type-safe queries
- **Authentication**: Clerk for user management and multi-tenant auth
- **API**: tRPC for end-to-end type safety and React Query integration
- **UI Components**: Radix UI primitives with custom styling
- **Styling**: Tailwind CSS 4+ with CSS-in-JS components
- **Charts**: Recharts for analytics and data visualization
- **AI Integration**: Vercel AI SDK for streaming chat responses
- **Payments**: Stripe for subscription and credit management

## Project Structure

```
adaptive-app/
├── src/
│   ├── app/                          # Next.js App Router pages
│   │   ├── _components/             # Page-specific components
│   │   │   ├── api-platform/        # API platform dashboard components
│   │   │   ├── chat-platform/       # Chat interface components
│   │   │   ├── landing_page/        # Marketing page components
│   │   │   └── stripe/              # Payment components
│   │   ├── api-platform/           # API platform pages
│   │   ├── chat-platform/          # Chat platform pages
│   │   ├── api/                    # API routes
│   │   │   ├── trpc/               # tRPC API handlers
│   │   │   ├── chat/               # Chat completion proxy
│   │   │   ├── stripe-checkout/    # Stripe webhook handlers
│   │   │   └── v1/                 # OpenAI-compatible API
│   │   ├── layout.tsx              # Root layout with providers
│   │   └── page.tsx                # Landing page
│   ├── components/
│   │   ├── ui/                     # Reusable UI components
│   │   │   ├── chat/               # Chat-specific components
│   │   │   └── *                   # Radix UI components
│   │   └── magicui/                # Enhanced UI components
│   ├── server/
│   │   ├── api/                    # tRPC routers and procedures
│   │   │   ├── routers/            # Individual API routers
│   │   │   ├── root.ts             # Root tRPC router
│   │   │   └── trpc.ts             # tRPC configuration
│   │   └── db.ts                   # Database connection
│   ├── lib/                        # Utilities and configurations
│   ├── hooks/                      # Custom React hooks
│   ├── types/                      # TypeScript type definitions
│   ├── context/                    # React context providers
│   ├── middleware.ts               # Clerk authentication middleware
│   └── env.js                      # Environment variable validation
├── prisma/
│   ├── schema.prisma              # Database schema
│   ├── migrations/                # Database migrations
│   └── generated/                 # Generated Prisma client
├── public/                        # Static assets
├── scripts/                       # Utility scripts
├── package.json                   # Dependencies and scripts
├── next.config.js                 # Next.js configuration
├── tailwind.config.js             # Tailwind CSS configuration
├── biome.jsonc                    # Code formatting configuration
└── tsconfig.json                  # TypeScript configuration
```

## Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/adaptive"
DIRECT_URL="postgresql://user:password@localhost:5432/adaptive"

# Authentication (Clerk)
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="pk_test_..."
CLERK_SECRET_KEY="sk_test_..."
NEXT_PUBLIC_CLERK_SIGN_IN_URL="/sign-in"
NEXT_PUBLIC_CLERK_SIGN_UP_URL="/sign-up"

# Stripe Payments
STRIPE_SECRET_KEY="sk_test_..."
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_..."
STRIPE_WEBHOOK_SECRET="whsec_..."

# Redis Cache
REDIS_URL="redis://localhost:6379"

# Backend Services
NEXT_PUBLIC_ADAPTIVE_API_URL="http://localhost:8080"
ADAPTIVE_AI_URL="http://localhost:8000"
```

### Optional Environment Variables

```bash
# Analytics
NEXT_PUBLIC_VERCEL_ANALYTICS_ID="..."

# Development
NODE_ENV="development"
ANALYZE="false"
```

## Development Commands

### Local Development
```bash
# Install dependencies
bun install

# Start development server with Turbo
bun dev

# Start development server (standard)
bun run dev

# Generate Prisma client and start dev
bun prisma generate && bun dev
```

### Database Management
```bash
# Generate Prisma client and run migrations
bun run db:generate

# Push schema changes to database
bun run db:push

# Open Prisma Studio database browser
bun run db:studio

# Seed provider data
bun run db:seed-providers

# Deploy migrations to production
bun run db:migrate
```

### Code Quality
```bash
# Run Biome linter and formatter
bun run check

# Auto-fix issues with Biome
bun run check:write

# Unsafe auto-fix (use with caution)
bun run check:unsafe

# TypeScript type checking
bun run typecheck
```

### Build and Deploy
```bash
# Build for production
bun run build

# Start production server
bun start

# Preview build locally
bun run preview
```

### Stripe Development
```bash
# Start Stripe webhook listener
bun run stripe

# Listen to specific events
stripe listen --events payment_intent.succeeded,customer.subscription.updated --forward-to localhost:3000/api/stripe-checkout
```

## Core Features

### Multi-Tenant Architecture

**Organizations → Projects → API Keys**
- Users can create and manage multiple organizations
- Each organization can have multiple projects
- Each project can have multiple API keys with different permissions
- Role-based access control (owner, admin, member)

### Chat Platform
- **Real-time streaming**: Server-sent events for streaming responses
- **Conversation management**: Save, organize, and search conversations
- **Model selection**: Choose from available providers and models
- **Cost tracking**: Real-time cost calculation and display
- **Message actions**: Copy, regenerate, edit messages

### API Platform Dashboard
- **Usage analytics**: Request counts, token usage, cost analysis
- **Provider comparison**: Performance and cost metrics across providers
- **Error monitoring**: Track API errors and success rates
- **Performance metrics**: Latency, throughput, and efficiency tracking
- **Credit management**: Monitor and top up organization credits

### Authentication & Security
- **Clerk Integration**: Social logins, email/password, multi-factor auth
- **API Key Security**: Secure generation with prefix identification
- **Rate Limiting**: Per-key rate limiting with Redis
- **CORS Protection**: Configurable origins and headers
- **Input Validation**: Zod schemas for all API inputs

## Database Schema

### Core Tables
- **organizations**: Multi-tenant organization management
- **projects**: Project organization within accounts
- **api_keys**: API key management with permissions
- **usage_records**: Detailed usage tracking and analytics
- **conversations**: Chat conversation persistence
- **messages**: Individual message storage with metadata
- **credits**: Organization credit tracking and billing

### Key Relationships
```sql
organizations (1) → (many) projects
projects (1) → (many) api_keys
projects (1) → (many) conversations
conversations (1) → (many) messages
organizations (1) → (many) credits
```

## API Routes

### tRPC Procedures
- **organizations**: CRUD operations for organizations
- **projects**: Project management within organizations
- **api_keys**: API key generation and management
- **conversations**: Chat conversation operations
- **messages**: Message CRUD with real-time updates
- **usage**: Analytics and usage tracking
- **credits**: Credit balance and transaction management

### REST API Routes
- **POST /api/chat**: Streaming chat completions
- **POST /api/v1/chat/completions**: OpenAI-compatible endpoint
- **POST /api/stripe-checkout**: Stripe webhook handler
- **GET /api/subscription-status**: Subscription status check

## UI Components

### Chat Components
- **ChatContainer**: Main chat interface with streaming
- **ChatMessages**: Message list with virtualization
- **ChatForm**: Input form with file upload support
- **MessageActions**: Copy, regenerate, edit functionality
- **CostComparison**: Real-time cost analysis display

### Dashboard Components
- **MetricCard**: Key performance indicator displays
- **UsageChart**: Time-series usage visualization
- **ProviderComparison**: Provider performance comparison
- **CreditManagement**: Credit balance and top-up interface
- **APIKeyManager**: API key generation and management

### UI System
- **Design System**: Consistent spacing, typography, and colors
- **Theme Support**: Dark/light mode with system preference
- **Responsive**: Mobile-first responsive design
- **Accessibility**: ARIA labels, keyboard navigation, screen reader support

## Performance Optimizations

### React Optimizations
- **Server Components**: Leverage React 19 server components
- **Streaming**: Progressive rendering with Suspense boundaries
- **Code Splitting**: Route-based and component-based splitting
- **Image Optimization**: Next.js Image component with optimization

### Database Optimizations
- **Connection Pooling**: Prisma connection pooling
- **Query Optimization**: Efficient queries with proper indexing
- **Caching**: Redis caching for frequently accessed data
- **Pagination**: Cursor-based pagination for large datasets

### Bundle Optimizations
- **Tree Shaking**: Remove unused code
- **Bundle Analysis**: Analyze bundle size with webpack-bundle-analyzer
- **Dynamic Imports**: Lazy load heavy components
- **Asset Optimization**: Optimize images, fonts, and static assets

## Security Considerations

### Authentication Security
- **JWT Validation**: Secure JWT token validation with Clerk
- **Session Management**: Secure session handling and rotation
- **CSRF Protection**: Built-in Next.js CSRF protection
- **XSS Prevention**: Content Security Policy headers

### API Security
- **Input Validation**: Comprehensive input validation with Zod
- **Rate Limiting**: API rate limiting per user/key
- **SQL Injection**: Prisma ORM prevents SQL injection
- **Sensitive Data**: Environment variable validation and encryption

### Data Protection
- **API Key Masking**: Display only key prefixes in UI
- **Audit Logging**: Track all administrative actions
- **Data Encryption**: Encrypt sensitive data at rest
- **GDPR Compliance**: User data deletion and export capabilities

## Deployment

### Production Build
```bash
# Build optimized production bundle
bun run build

# Start production server
bun start

# Environment variables validation
# Verify all required environment variables are set
```

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app

# Install dependencies
COPY package.json bun.lock ./
RUN npm install -g bun && bun install --frozen-lockfile

# Copy source code
COPY . .

# Generate Prisma client and build
RUN bun prisma generate && bun run build

EXPOSE 3000
CMD ["bun", "start"]
```

### Vercel Deployment
- **Automatic Deployments**: Connect GitHub repository for auto-deployment
- **Environment Variables**: Configure production environment variables
- **Database**: Use production PostgreSQL database
- **Analytics**: Enable Vercel Analytics for performance monitoring

## Monitoring & Analytics

### Performance Monitoring
- **Core Web Vitals**: Track LCP, FID, CLS metrics
- **Custom Metrics**: Track chat response times and API latency
- **Error Tracking**: Monitor and alert on application errors
- **Usage Analytics**: Track feature usage and user behavior

### Business Metrics
- **User Engagement**: Track active users and session duration
- **Cost Metrics**: Monitor API costs and credit consumption
- **Conversion Rates**: Track trial-to-paid conversion
- **Feature Adoption**: Monitor feature usage and adoption rates

## Troubleshooting

### Common Issues

**Build failures**
- Run `bun run typecheck` to identify TypeScript errors
- Check environment variable configuration
- Verify Prisma schema and generate client: `bun prisma generate`
- Clear Next.js cache: `rm -rf .next`

**Database connection errors**
- Verify DATABASE_URL format and credentials
- Check database server connectivity
- Run database migrations: `bun run db:generate`
- Check Prisma schema for syntax errors

**Authentication issues**
- Verify Clerk environment variables
- Check middleware configuration in `middleware.ts`
- Verify authentication routes and redirects
- Check browser cookies and local storage

**API errors**
- Verify backend service connectivity (adaptive-backend, adaptive_ai)
- Check CORS configuration for API calls
- Verify tRPC router configuration
- Check network connectivity and firewall settings

### Debug Commands
```bash
# Check TypeScript errors
bun run typecheck

# Lint and format code
bun run check

# Debug database schema
bun prisma studio

# Check environment variables
bun -e "console.log(process.env)"

# Analyze bundle size
ANALYZE=true bun run build

# Check dependency issues
bun install --frozen-lockfile
```

## Contributing

### Code Style
- **Formatting**: Biome for consistent code formatting
- **Type Safety**: Strict TypeScript configuration
- **Component Structure**: Consistent component organization and naming
- **Import Organization**: Absolute imports with path mapping
- **Error Boundaries**: Comprehensive error handling

### Testing Requirements
- **Unit Tests**: Component and utility function tests
- **Integration Tests**: API route and database interaction tests
- **E2E Tests**: Critical user flow testing
- **Accessibility Tests**: ARIA compliance and keyboard navigation
- **Performance Tests**: Core Web Vitals and load testing

### Documentation Updates
**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new pages, components, or features
   - Modifying database schema or API routes
   - Changing environment variables or configuration
   - Adding new dependencies or updating major versions
   - Modifying authentication or security implementations
   - Adding new tRPC procedures or modifying existing ones

2. **Update root CLAUDE.md** when:
   - Changing service ports, commands, or basic service description
   - Modifying the service's role in the overall architecture
   - Adding new external integrations or dependencies

3. **Update README and component documentation** when:
   - Adding new UI components or modifying existing ones
   - Changing user workflows or interface patterns
   - Adding new features that affect user experience

### Pull Request Process
1. Create feature branch from `dev`
2. Implement changes with comprehensive tests
3. Run quality checks: `bun run typecheck && bun run check`
4. **Update relevant documentation** (CLAUDE.md files, component docs, README)
5. Submit PR with clear description, screenshots, and documentation updates
6. Ensure all CI checks pass including build, lint, and type checking