# Adaptive JavaScript SDK

JavaScript/TypeScript client library for the Adaptive AI platform - drop-in replacement for OpenAI with intelligent model selection.

## Installation

```bash
npm install adaptive-js
# or
yarn add adaptive-js
# or
pnpm add adaptive-js
```

## Quick Start

```typescript
import { Adaptive } from 'adaptive-js';

// Initialize client
const client = new Adaptive({
  apiKey: 'your-api-key'
});

// Create chat completion (automatically selects optimal model)
const response = await client.chat
.completions.create({
  messages: [
    { role: 'user', content: 'Write a JavaScript function to sort an array' }
  ]
});

console.log(response.choices[0].message.content);
```

## Features

- **Drop-in OpenAI Replacement**: Compatible with OpenAI JavaScript SDK interface
- **Intelligent Model Selection**: Automatic routing to optimal models
- **Multi-Provider Support**: Access OpenAI, Anthropic, Groq, DeepSeek models
- **Streaming Support**: Real-time response streaming
- **TypeScript First**: Full TypeScript support with type definitions
- **Modern Standards**: ES modules, async/await, fetch-based

## Configuration

### Environment Variables

```bash
ADAPTIVE_API_KEY=your-api-key
ADAPTIVE_BASE_URL=https://api.adaptive.ai  # Optional
```

### Programmatic Configuration

```typescript
import { Adaptive } from 'adaptive-js';

const client = new Adaptive({
  apiKey: 'your-api-key',
  baseURL: 'https://api.adaptive.ai'  // Optional
});
```

## Usage Examples

### Basic Chat Completion

```typescript
const response = await client.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain quantum computing' }
  ]
});

console.log(response.choices[0].message.content);
```

### Streaming Chat Completion

```typescript
const stream = await client.chat.completions.create({
  messages: [{ role: 'user', content: 'Tell me a story' }],
  stream: true
});

for await (const chunk of stream) {
  if (chunk.choices[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}
```

### Error Handling

```typescript
try {
  const response = await client.chat.completions.create({
    messages: [{ role: 'user', content: 'Hello' }]
  });
} catch (error) {
  if (error.status === 401) {
    console.error('Invalid API key');
  } else {
    console.error('API error:', error.message);
  }
}
```

## API Reference

### Chat Completions

#### `client.chat.completions.create()`

**Parameters:**
- `messages` (Array): Array of message objects
- `stream` (boolean, optional): Enable streaming responses
- `max_tokens` (number, optional): Maximum tokens in response
- `temperature` (number, optional): Sampling temperature (0-2)

**Returns:**
- Promise resolving to `ChatCompletion` object

### Type Definitions

```typescript
interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  provider: string;  // Added by Adaptive
  choices: Choice[];
  usage: Usage;
}

interface Choice {
  index: number;
  message: Message;
  finish_reason: string;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
```

## Migration from OpenAI

Replace OpenAI imports with Adaptive:

```typescript
// Before
import OpenAI from 'openai';
const client = new OpenAI({ apiKey: 'sk-...' });

// After
import { Adaptive } from 'adaptive-js';
const client = new Adaptive({ apiKey: 'your-adaptive-key' });

// Same interface!
const response = await client.chat.completions.create({...});
```

## Browser Support

Works in modern browsers and Node.js environments:

```html
<!-- ES Modules -->
<script type="module">
  import { Adaptive } from 'https://unpkg.com/adaptive-js/dist/index.js';
  
  const client = new Adaptive({ apiKey: 'your-key' });
  // Use client...
</script>
```

## Development

```bash
# Install dependencies
bun install

# Build
bun run build

# Test
bun test

# Type check
bun run typecheck
```

## Examples

- **Node.js**: Basic server-side usage
- **React**: Frontend chat application
- **Streaming**: Real-time response handling
- **Error Handling**: Robust error management

## Requirements

- Node.js 16+ or modern browser
- ES2020+ support

## License

MIT License - see LICENSE file for details.