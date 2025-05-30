# Adaptive Frontend

React application providing the user interface for the Adaptive AI platform with real-time chat and dashboard features.

## Features

- **Interactive Chat**: Real-time streaming chat with syntax highlighting
- **Dashboard**: Usage analytics and conversation management
- **API Key Management**: Create and manage access keys
- **Authentication**: Clerk integration with social logins
- **Responsive Design**: Mobile-optimized with dark/light themes
- **Modern Stack**: React 18, TypeScript, TanStack Router, Tailwind CSS

## Quick Start

```bash
# Install dependencies
pnpm install

# Set environment variables
cp .env.example .env.local

# Start development server
pnpm dev
```

## Environment Variables

```bash
VITE_CLERK_PUBLISHABLE_KEY=pk_test_xxxxx
VITE_BASE_API_URL=http://localhost:8080
```

## Available Scripts

```bash
pnpm dev          # Start development server
pnpm build        # Build for production
pnpm preview      # Preview production build
pnpm typecheck    # Type checking
pnpm lint         # Lint code
```

## Project Structure

```
src/
├── components/           # Reusable UI components
│   ├── ui/              # Base components (Radix UI)
│   ├── chat/            # Chat interface components
│   └── landing_page/    # Landing page sections
├── pages/               # Page components
├── services/            # API service layer
├── hooks/               # Custom React hooks
├── routes/              # TanStack Router definitions
└── lib/                 # Utility functions
```

## Key Technologies

- **React 18** - UI framework with concurrent features
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **TanStack Router** - Type-safe routing
- **Tailwind CSS** - Utility-first styling
- **Clerk** - Authentication
- **Zustand** - State management
- **React Query** - Server state management
- **Radix UI** - Accessible components

## Building

```bash
# Development build
pnpm build

# Docker build
docker build -t adaptive-frontend .
```

## Testing

```bash
pnpm test
```