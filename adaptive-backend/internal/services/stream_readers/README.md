# Stream Readers Module

A flexible streaming response system for handling real-time chat completions from multiple LLM providers. This module provides adapters for converting provider-specific streaming formats into standardized outputs compatible with both Server-Sent Events (SSE) and Vercel AI SDK DataStream protocols.

## Overview

The stream_readers module serves as the streaming layer between LLM providers and client applications. It abstracts the complexity of different provider streaming APIs and formats them into consistent, client-friendly protocols. The module supports two primary streaming formats:

- **Server-Sent Events (SSE)** - Standard web streaming protocol
- **Vercel DataStream** - Optimized format for Vercel AI SDK integration

## Architecture

```
stream_readers/
├── stream_readers.go          # Core interfaces and base implementation
├── stream/
│   └── stream.go             # Main streaming orchestration
├── sse/
│   ├── sse.go               # SSE factory and router
│   ├── openai.go            # OpenAI SSE stream reader
│   ├── anthropic.go         # Anthropic SSE stream reader
│   ├── groq.go              # Groq SSE stream reader
│   └── deepseek.go          # DeepSeek SSE stream reader
└── vercel/
    ├── datastream.go        # DataStream factory and router
    ├── vercel.go            # Core DataStream implementation
    ├── adapter.go           # Provider adapter interface
    ├── openai.go            # OpenAI DataStream adapter
    ├── anthropic.go         # Anthropic DataStream adapter
    ├── groq.go              # Groq DataStream adapter
    └── deepseek.go          # DeepSeek DataStream adapter
```

## Core Interfaces

### StreamReader Interface

```go
type StreamReader interface {
    io.Reader
    io.Closer
}
```

### BaseStreamReader

```go
type BaseStreamReader struct {
    Buffer    []byte
    RequestID string
    CloseLock sync.Once
}
```

## Streaming Protocols

### Server-Sent Events (SSE)

SSE format follows the standard web specification for real-time communication:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","provider":"openai",...}

data: {"choices":[{"delta":{"content":"Hello"}}],"provider":"openai"}

data: [DONE]

```

#### SSE Enhanced Response Format

Each provider's response is enhanced with provider identification:

```go
type EnhancedOpenAIResponse struct {
    openai.ChatCompletionChunk
    Provider string `json:"provider"`
}
```

### Vercel DataStream

DataStream format is optimized for the Vercel AI SDK:

```
0:"Hello"

0:" world"

d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5}}

```

#### DataStream Protocol Elements

- `0:"text"` - Text content chunks
- `1:{}` - Function call data (reserved)
- `2:{}` - Tool call data (reserved)  
- `3:"error"` - Error messages
- `d:{}` - Final data with finish reason and usage

## Provider Implementations

### OpenAI Stream Reader

**SSE Implementation:**
```go
type OpenAIStreamReader struct {
    stream_readers.BaseStreamReader
    stream *ssestream.Stream[openai.ChatCompletionChunk]
    done   bool
}
```

**Features:**
- Native OpenAI SSE stream handling
- Automatic error formatting
- Multiple choice support
- Proper finish reason detection

### Anthropic Stream Reader

**SSE Implementation:**
```go
type AnthropicStreamReader struct {
    stream_readers.BaseStreamReader
    stream ssestream.Stream[anthropic.MessageStreamEventUnion]
    done   bool
}
```

**Features:**
- Event-based streaming (ContentBlockDelta, MessageStop)
- System prompt handling
- Text delta processing
- Enhanced error reporting

### Groq Stream Reader

**SSE Implementation:**
```go
type GroqStreamReader struct {
    stream_readers.BaseStreamReader
    stream *groq.ChatCompletionStream
    done   bool
}
```

**Features:**
- OpenAI-compatible streaming
- High-throughput optimization
- Low-latency response handling
- Finish reason detection

### DeepSeek Stream Reader

**SSE Implementation:**
```go
type DeepSeekStreamReader struct {
    stream_readers.BaseStreamReader
    stream deepseek.ChatCompletionStream
    done   bool
}
```

**Features:**
- OpenAI-compatible format
- Delta content processing
- Stream termination handling
- Error propagation

## Adapter Pattern for DataStream

### Provider Adapter Interface

```go
type ProviderAdapter interface {
    NextChunk() (*InternalProviderChunk, error)
    io.Closer
}
```

### Internal Chunk Format

```go
type InternalProviderChunk struct {
    Text         string `json:"text,omitempty"`
    ToolCalls    []any  `json:"tool_calls,omitempty"`
    Error        string `json:"error,omitempty"`
    FinishReason string `json:"finish_reason,omitempty"`
}
```

## Usage Examples

### Basic Stream Handling

```go
import (
    "adaptive-backend/internal/services/stream_readers/stream"
    "github.com/gofiber/fiber/v2"
)

func HandleChatStream(c *fiber.Ctx) error {
    // Get chat completion response from provider
    resp, err := provider.StreamChatCompletion(req)
    if err != nil {
        return err
    }
    
    // Handle streaming with automatic format detection
    return stream.HandleStream(c, resp, requestID, options)
}
```

### SSE Stream Creation

```go
import "adaptive-backend/internal/services/stream_readers/sse"

// Create SSE stream reader
streamReader, err := sse.GetSSEStreamReader(resp, "openai", requestID)
if err != nil {
    return err
}
defer streamReader.Close()

// Read streaming data
buffer := make([]byte, 1024)
for {
    n, err := streamReader.Read(buffer)
    if err == io.EOF {
        break
    }
    // Process buffer[:n]
}
```

### DataStream Creation

```go
import "adaptive-backend/internal/services/stream_readers/vercel"

// Create DataStream reader
streamReader, err := vercel.GetDataStreamReader(resp, "anthropic", requestID)
if err != nil {
    return err
}
defer streamReader.Close()

// Stream data to Vercel AI SDK
// (handled automatically by stream.HandleStream)
```

### Manual Stream Processing

```go
// Using provider adapter directly
adapter := vercel.NewOpenAIAdapter(stream, requestID)
defer adapter.Close()

for {
    chunk, err := adapter.NextChunk()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Printf("Stream error: %v", err)
        break
    }
    
    if chunk.Text != "" {
        fmt.Printf("Received: %s", chunk.Text)
    }
}
```

## Request Options

### Stream Selection

```go
type RequestOptions struct {
    StreamOptions *StreamOption `json:"stream_options,omitempty"`
}

type StreamOption string

const (
    SSE        StreamOption = "sse"        // Server-Sent Events (default)
    DATASTREAM StreamOption = "datastream" // Vercel AI SDK format
)
```

### Usage in Requests

```go
options := &models.RequestOptions{
    StreamOptions: &models.DATASTREAM,
}

err := stream.HandleStream(c, resp, requestID, options)
```

## Error Handling

### Consistent Error Format

All stream readers provide consistent error handling:

```go
// SSE Error Format
data: {"error": "Stream connection failed"}

// DataStream Error Format  
3:"Stream connection failed"
```

### Error Types

- **Connection Errors**: Network failures, timeouts
- **Provider Errors**: API errors, rate limits, authentication
- **Format Errors**: JSON marshaling, protocol violations
- **Stream Errors**: Unexpected stream termination

### Error Recovery

```go
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
    // Handle provider stream errors
    if !r.stream.Next() {
        if r.stream.Err() != nil {
            // Format error as SSE and terminate
            safeErrMsg := strings.ReplaceAll(r.stream.Err().Error(), "\"", "\\\"")
            r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
            r.done = true
        }
    }
}
```

## Performance Optimization

### Buffer Management

- **Buffer Pooling**: Reuse buffers to reduce allocations
- **Chunk Size**: Optimal 1KB chunks for streaming
- **Memory Limits**: Bounded buffer growth

### Streaming Efficiency

```go
// Efficient streaming loop
func pumpStreamData(w *bufio.Writer, streamReader io.Reader, requestID string) error {
    buffer := make([]byte, 1024) // Fixed buffer size
    
    for {
        n, err := streamReader.Read(buffer)
        if n > 0 {
            if err := writeAndFlush(w, buffer[:n], requestID); err != nil {
                return err
            }
        }
        if err == io.EOF {
            return nil
        }
        if err != nil {
            return err
        }
    }
}
```

### Connection Management

- **Graceful Closure**: Proper resource cleanup
- **Timeout Handling**: Request timeout management
- **Concurrent Safety**: Thread-safe stream operations

## Monitoring and Logging

### Request Tracking

```go
log.Printf("[%s] Stream started for provider: %s", requestID, provider)
log.Printf("[%s] Stream completed after %v", requestID, duration)
```

### Error Logging

```go
log.Printf("[%s] Error in %s stream: %v", requestID, provider, err)
log.Printf("[%s] Failed to write stream data: %v", requestID, err)
```

### Performance Metrics

- Stream duration tracking
- Error rate monitoring
- Provider-specific metrics
- Buffer utilization stats

## Testing

### Unit Tests

```go
func TestOpenAIStreamReader(t *testing.T) {
    // Create mock stream
    mockStream := createMockOpenAIStream()
    reader := NewOpenAIStreamReader(mockStream, "test-123")
    
    // Test reading
    buffer := make([]byte, 1024)
    n, err := reader.Read(buffer)
    
    assert.NoError(t, err)
    assert.Greater(t, n, 0)
    assert.Contains(t, string(buffer[:n]), "data:")
}
```

### Integration Tests

```go
func TestProviderStreaming(t *testing.T) {
    providers := []string{"openai", "anthropic", "groq", "deepseek"}
    
    for _, provider := range providers {
        t.Run(provider, func(t *testing.T) {
            // Test both SSE and DataStream formats
            testStreamingFormat(t, provider, "sse")
            testStreamingFormat(t, provider, "datastream")
        })
    }
}
```

### Mock Implementations

```go
type MockProviderAdapter struct {
    chunks []InternalProviderChunk
    index  int
}

func (m *MockProviderAdapter) NextChunk() (*InternalProviderChunk, error) {
    if m.index >= len(m.chunks) {
        return nil, io.EOF
    }
    chunk := m.chunks[m.index]
    m.index++
    return &chunk, nil
}
```

## Future Enhancements

### Planned Features

- **WebSocket Support**: Real-time bidirectional communication
- **Compression**: Stream compression for bandwidth optimization
- **Multiplexing**: Multiple concurrent streams per connection
- **Quality of Service**: Priority-based stream handling

### Protocol Extensions

- **Custom Events**: Application-specific event types
- **Metadata Streaming**: Rich metadata alongside content
- **Binary Streaming**: Support for non-text content
- **Backpressure**: Flow control for high-volume streams

## Dependencies

```go
import (
    // Core streaming
    "bufio"
    "io"
    "sync"
    
    // Provider SDKs
    "github.com/openai/openai-go"
    "github.com/anthropics/anthropic-sdk-go"
    "github.com/conneroisu/groq-go"
    "github.com/cohesion-org/deepseek-go"
    
    // Web framework
    "github.com/gofiber/fiber/v2"
    "github.com/valyala/fasthttp"
)
```

## Contributing

When adding support for new providers:

1. **SSE Implementation**: Create provider-specific SSE reader
2. **DataStream Adapter**: Implement ProviderAdapter interface
3. **Factory Integration**: Add provider to routing functions
4. **Error Handling**: Implement consistent error formatting
5. **Testing**: Add comprehensive unit and integration tests
6. **Documentation**: Update this README with provider details

### Code Guidelines

- Follow existing naming conventions
- Implement both SSE and DataStream support
- Include proper error handling and logging
- Add request ID tracking for debugging
- Ensure thread-safe operations
- Handle resource cleanup properly

## License

This module is part of the Adaptive Backend project and follows the same licensing terms.