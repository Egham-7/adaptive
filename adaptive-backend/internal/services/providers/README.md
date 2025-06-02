# LLM Providers Module

A unified interface for multiple Large Language Model (LLM) providers in the Adaptive Backend. This module provides a consistent API across different LLM services, enabling seamless switching between providers and supporting both streaming and non-streaming chat completions.

## Overview

The providers module abstracts the complexity of integrating with different LLM providers, each with their own APIs, authentication methods, and response formats. It provides a unified interface that allows the rest of the application to work with any provider without knowing the implementation details.

## Supported Providers

| Provider | Service Class | Official SDK | Environment Variable |
|----------|---------------|--------------|---------------------|
| OpenAI | `OpenAIService` | ✅ | `OPENAI_API_KEY` |
| Anthropic | `AnthropicService` | ✅ | `ANTHROPIC_API_KEY` |
| Groq | `GroqService` | ✅ | `GROQ_API_KEY` |
| DeepSeek | `DeepSeekService` | ✅ | `DEEPSEEK_API_KEY` |
| Google Gemini | `GeminiService` | ✅ | `GOOGLE_API_KEY` |

## Architecture

### Core Interface

All providers implement the `LLMProvider` interface:

```go
type LLMProvider interface {
    CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error)
    StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error)
}
```

### Factory Pattern

The module uses a factory pattern for provider instantiation:

```go
func NewLLMProvider(providerName string) (LLMProvider, error)
```

### Request/Response Flow

```
Client Request → Provider Router → Provider Factory → Specific Provider → Provider API → Unified Response
```

## Provider Implementations

### OpenAI Service (`openai_service.go`)

**Features:**
- Full support for GPT-3.5, GPT-4, and GPT-4o models
- Function calling and tool use
- Image inputs and vision capabilities
- Streaming and non-streaming responses

**Supported Models:**
- GPT-4o (default)
- GPT-4.1
- GPT-4.1 Mini
- O3, O4 Mini
- GPT-4.1 Nano

### Anthropic Service (`anthropic_service.go`)

**Features:**
- Claude 3 (Opus, Sonnet, Haiku) model support
- System prompts and conversation management
- Tool use and function calling
- Streaming responses with proper event handling

**Supported Models:**
- Claude 3.5 Haiku Latest
- Claude 3.5 Sonnet Latest
- Claude 4 Sonnet (2025-05-14)
- Claude 4 Opus (2025-05-14)
- Claude 3.7 Sonnet Latest (default)

### Groq Service (`groq_service.go`)

**Features:**
- Ultra-fast inference with specialized hardware
- Mixtral, Llama, and Gemma model support
- High-throughput streaming
- Low-latency responses

**Supported Models:**
- Llama 3.2 3B Preview (default)
- Llama 3.1 70B Versatile
- Llama 3.1 8B Instant
- Llama 3.2 90B Vision
- Llama 3.3 70B Specdec
- Gemma 7B, Gemma 2 9B

### DeepSeek Service (`deepseek_service.go`)

**Features:**
- DeepSeek Code and Chat models
- Multi-language programming support
- Cost-effective inference
- Streaming and batch processing

**Supported Models:**
- DeepSeek Chat (default)
- DeepSeek Reasoner

### Google Gemini Service (`gemini_service.go`)

**Features:**
- Gemini Pro and Gemini Pro Vision
- Multimodal inputs
 (text, images)
- Large context windows
- Google AI integration

## Configuration

### Environment Variables

Each provider requires its respective API key to be set as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### Usage Example

```go
// Initialize a provider
provider, err := providers.NewLLMProvider("openai")
if err != nil {
    log.Fatal(err)
}

// Create a chat completion request
req := &models.ProviderChatCompletionRequest{
    Model: "gpt-4o",
    Messages: []models.Message{
        {Role: "user", Content: "Hello, how are you?"},
    },
    MaxTokens: 100,
    Temperature: 0.7,
}

// Get response
resp, err := provider.CreateChatCompletion(req)
if err != nil {
    log.Fatal(err)
}
```

### Streaming Example

```go
// Create streaming request
req := &models.ProviderChatCompletionRequest{
    Model: "gpt-4o",
    Messages: []models.Message{
        {Role: "user", Content: "Tell me a story"},
    },
    Stream: true,
}

// Get streaming response
resp, err := provider.StreamChatCompletion(req)
if err != nil {
    log.Fatal(err)
}

// Handle the stream response
// (Implementation depends on provider-specific stream types)
```

## Request Parameters

| Parameter | Type | Description | Supported Providers |
|-----------|------|-------------|-------------------|
| `Model` | string | Model identifier | All providers |
| `Messages` | []Message | Conversation history | All providers |
| `Temperature` | float32 | Randomness (0.0-1.0) | All providers |
| `TopP` | float32 | Nucleus sampling | Most providers |
| `MaxTokens` | int | Maximum response tokens | All providers |
| `PresencePenalty` | float32 | Presence penalty | OpenAI, Groq, DeepSeek |
| `FrequencyPenalty` | float32 | Frequency penalty | OpenAI, Groq, DeepSeek |
| `Stream` | bool | Enable streaming | All providers |

## Request/Response Models

### Unified Request Format

```go
type ProviderChatCompletionRequest struct {
    Messages         []Message               `json:"messages"`
    Model            string                  `json:"model,omitempty"`
    MaxTokens        *int                    `json:"max_tokens,omitempty"`
    Temperature      *float64                `json:"temperature,omitempty"`
    TopP             *float64                `json:"top_p,omitempty"`
    Stream           bool                    `json:"stream,omitempty"`
    Stop             []string                `json:"stop,omitempty"`
    Tools            []Tool                  `json:"tools,omitempty"`
    ToolChoice       interface{}             `json:"tool_choice,omitempty"`
    ResponseFormat   *ResponseFormat         `json:"response_format,omitempty"`
    Seed             *int                    `json:"seed,omitempty"`
    User             string                  `json:"user,omitempty"`
    
    // Provider-specific fields
    SystemPrompt     string                  `json:"system_prompt,omitempty"`     // Anthropic
    SafetySettings   []SafetySetting         `json:"safety_settings,omitempty"`   // Gemini
    RepetitionPenalty *float64               `json:"repetition_penalty,omitempty"` // DeepSeek
}
```

### Unified Response Format

```go
type ChatCompletionResponse struct {
    ID                string                 `json:"id"`
    Object            string                 `json:"object"`
    Created           int64                  `json:"created"`
    Model             string                 `json:"model"`
    Provider          string                 `json:"provider"`
    Choices           []Choice               `json:"choices"`
    Usage             *Usage                 `json:"usage,omitempty"`
    SystemFingerprint string                 `json:"system_fingerprint,omitempty"`
    
    // Streaming support
    Stream            io.ReadCloser          `json:"-"`
}
```

## Error Handling

### Provider-Specific Errors

Each provider handles specific error conditions:

```go
type ProviderError struct {
    Provider    string
    Type        ErrorType
    Message     string
    StatusCode  int
    Retryable   bool
    RetryAfter  time.Duration
}

type ErrorType int

const (
    ErrorTypeAuth ErrorType = iota
    ErrorTypeRateLimit
    ErrorTypeInvalidRequest
    ErrorTypeServerError
    ErrorTypeNetwork
    ErrorTypeTimeout
)
```

### Retry Logic

```go
func (p *BaseProvider) WithRetry(operation func() error) error {
    backoff := time.Second
    
    for attempt := 0; attempt < MaxRetries; attempt++ {
        err := operation()
        if err == nil {
            return nil
        }
        
        if !isRetryable(err) {
            return err
        }
        
        time.Sleep(backoff)
        backoff *= 2 // Exponential backoff
    }
    
    return errors.New("max retries exceeded")
}
```

## Testing

### Unit Tests

Each provider has comprehensive unit tests:

```go
func TestOpenAIService_CreateChatCompletion(t *testing.T) {
    tests := []struct {
        name           string
        request        *models.ProviderChatCompletionRequest
        mockResponse   string
        expectedError  error
        expectedModel  string
    }{
        {
            name: "successful completion",
            request: &models.ProviderChatCompletionRequest{
                Messages: []models.Message{
                    {Role: "user", Content: "Hello"},
                },
            },
            mockResponse:  validOpenAIResponse,
            expectedError: nil,
            expectedModel: "gpt-3.5-turbo",
        },
        // More test cases...
    }
}
```

### Integration Tests

```go
func TestProviders_Integration(t *testing.T) {
    providers := []string{"openai", "anthropic", "groq", "deepseek", "gemini"}
    
    for _, providerName := range providers {
        t.Run(providerName, func(t *testing.T) {
            provider, err := NewLLMProvider(providerName)
            require.NoError(t, err)
            
            response, err := provider.CreateChatCompletion(testRequest)
            require.NoError(t, err)
            assert.NotEmpty(t, response.Choices[0].Message.Content)
        })
    }
}
```

### Mock Providers

```go
type MockProvider struct {
    responses []models.ChatCompletionResponse
    errors    []error
    callCount int
}

func (m *MockProvider) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
    if m.callCount < len(m.errors) && m.errors[m.callCount] != nil {
        return nil, m.errors[m.callCount]
    }
    
    if m.callCount < len(m.responses) {
        response := m.responses[m.callCount]
        m.callCount++
        return &response, nil
    }
    
    return nil, errors.New("no more mock responses")
}
```

## Performance Optimization

### Connection Pooling

```go
type ProviderConfig struct {
    MaxIdleConns        int
    MaxConnsPerHost     int
    IdleConnTimeout     time.Duration
    RequestTimeout      time.Duration
    TLSHandshakeTimeout time.Duration
}
```

### Request Optimization

- **Request Batching**: Batch multiple requests when supported
- **Compression**: Use gzip compression for request/response
- **Caching**: Cache responses when appropriate
- **Parallel Processing**: Process multiple providers concurrently

### Memory Management

- **Buffer Pooling**: Reuse buffers for streaming responses
- **Garbage Collection**: Minimize allocations and GC pressure
- **Resource Cleanup**: Properly clean up resources after use
- **Memory Limits**: Implement memory usage limits

## Monitoring and Observability

### Metrics

```go
var (
    requestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "llm_requests_total",
            Help: "Total number of LLM requests",
        },
        []string{"provider", "model", "status"},
    )
    
    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "llm_request_duration_seconds",
            Help: "Duration of LLM requests",
        },
        []string{"provider", "model"},
    )
)
```

### Logging

```go
logger.Info("LLM request completed",
    "provider", provider,
    "model", model,
    "duration_ms", duration.Milliseconds(),
    "tokens_used", usage.TotalTokens,
    "request_id", requestID,
)
```

## Configuration

### Environment Variables

```bash
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GOOGLE_AI_API_KEY=...

# Provider Settings
LLM_REQUEST_TIMEOUT=30s
LLM_MAX_RETRIES=3
LLM_RETRY_BACKOFF=1s
LLM_ENABLE_CACHING=true
LLM_CACHE_TTL=300s
```

### Provider Selection

```go
type ProviderSelector struct {
    DefaultProvider string
    ProviderWeights map[string]float64
    HealthChecks    map[string]func() bool
}
```

## Future Enhancements

### Planned Features

- **Provider Load Balancing**: Distribute load across provider instances
- **Automatic Failover**: Automatic failover between providers
- **Cost Optimization**: Intelligent routing based on cost
- **Quality Scoring**: Track and route based on response quality

### New Provider Support

- **Cohere**: Support for Cohere's Command models
- **AI21**: Integration with AI21's Jurassic models
- **Hugging Face**: Support for Hugging Face Inference API
- **Local Models**: Support for local model deployment

### Advanced Features

- **Model Routing**: Intelligent routing based on request characteristics
- **A/B Testing**: Support for provider A/B testing
- **Response Caching**: Intelligent caching based on request similarity
- **Cost Analytics**: Detailed cost tracking and optimization