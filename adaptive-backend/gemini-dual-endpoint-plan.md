# Gemini Dual Endpoint Implementation Plan

## Overview
Implement dual endpoint architecture to support both OpenAI-compatible format and native Gemini format responses in the adaptive-backend.

## Architecture Design

### Endpoint Structure
- **OpenAI Format**: `/v1/chat/completions` (existing)
- **Gemini Format**: `/v1/generateContent` (new)

### Key Principles
1. Keep existing OpenAI-compatible endpoint intact
2. Add separate Gemini-native endpoint for native response format
3. Use official Go Gemini SDK for native calls
4. Maintain provider abstraction layer
5. No response format conversion - each endpoint returns its native format

## Implementation Steps

### 1. Add ResponseFormat Field to Models
**File**: `internal/models/completions.go`
- Add `ResponseFormat string` field to `ChatCompletionRequest`
- Values: `"openai"`, `"gemini"`, `"anthropic"`
- Default to `"openai"` for backward compatibility

### 2. Create Gemini Provider Service
**New Directory**: `internal/services/providers/gemini/`
- Create `gemini.go` with GeminiService struct
- Use `google.golang.org/genai` SDK
- Implement native Gemini API calls
- Return `genai.GenerateContentResponse` directly

### 3. Update Provider Interfaces
**File**: `internal/services/providers/provider_interfaces/provider.go`
- Add `GeminiProvider` interface for native Gemini calls
- Keep existing `LLMProvider` interface for OpenAI-compatible calls
- Add methods for native response handling

### 4. Create Gemini Endpoint Handler
**New File**: `internal/api/gemini.go`
- Handle `/v1/generateContent` requests
- Use adaptive-ai service for model selection (same as OpenAI endpoint)
- Convert OpenAI request format to Gemini format
- Return native Gemini response structure
- Support both streaming and non-streaming

### 5. Update Provider Factory
**File**: `internal/services/providers/llm_provider.go`
- Add `NewGeminiProvider()` function
- Create provider selection logic based on endpoint type
- Maintain existing OpenAI-compatible provider creation

### 6. Add Routing Configuration
**File**: `cmd/api/main.go` or routing setup
- Register new Gemini endpoint route
- Ensure proper parameter extraction for model name
- Maintain existing OpenAI route

### 7. Update Configuration
**File**: `internal/config/config.go`
- Ensure Gemini provider config supports both endpoint types
- Keep existing YAML configuration structure
- No changes needed to provider config structure

## Technical Details

### Request Flow for OpenAI Endpoint
```
POST /v1/chat/completions
→ OpenAI Handler
→ OpenAI-compatible Provider
→ Provider-specific API call
→ Convert to OpenAI format
→ Return OpenAI ChatCompletion
```

### Request Flow for Gemini Endpoint
```
POST /v1/models/{model}:generateContent
→ Gemini Handler
→ Native Gemini Provider
→ Direct Gemini API call
→ Return native genai.GenerateContentResponse
```

### Provider Selection Logic
- **OpenAI Endpoint**: Use existing provider selection (OpenAI, Anthropic, etc.)
- **Gemini Endpoint**: Always use native Gemini provider
- **Response Format**: Determined by endpoint, not request parameter

## File Structure Changes

### New Files
```
internal/services/providers/gemini/
├── gemini.go                 # Native Gemini service
└── chat/
    └── gemini_chat.go       # Gemini chat implementation

internal/api/
└── gemini.go                # Gemini endpoint handler
```

### Modified Files
```
internal/models/completions.go           # Add ResponseFormat field
internal/services/providers/llm_provider.go  # Add Gemini provider factory
internal/services/providers/provider_interfaces/provider.go  # Add Gemini interface
cmd/api/main.go                         # Add Gemini route
```

## Dependencies

### New Go Module Dependencies
```go
google.golang.org/genai  // Official Go Gemini SDK
```

### Existing Dependencies (unchanged)
```go
github.com/openai/openai-go     // OpenAI SDK
github.com/gofiber/fiber/v2     // Web framework
```

## API Examples

### OpenAI Endpoint (existing)
```http
POST /v1/chat/completions
Content-Type: application/json
X-Stainless-API-Key: your-key

{
  "model": "gemini-2.0-flash",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}

Response: OpenAI ChatCompletion format
```

### Gemini Endpoint (new)
```http
POST /v1/models/gemini-2.0-flash:generateContent
Content-Type: application/json
X-Stainless-API-Key: your-key

{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}

Response: Native Gemini GenerateContentResponse format
```

## Configuration Impact

### YAML Config (no changes needed)
```yaml
providers:
  gemini:
    api_key: "${GEMINI_API_KEY}"
    base_url: ""  # Use SDK default for native calls
```

### Environment Variables (no changes needed)
```bash
GEMINI_API_KEY=your-gemini-key
```

## Testing Strategy

### Unit Tests
- Test native Gemini provider service
- Test Gemini endpoint handler
- Test request/response conversion
- Test provider factory selection

### Integration Tests
- Test both endpoints with real API calls
- Verify response format differences
- Test error handling for both endpoints
- Test streaming for both endpoints

## Migration Strategy

### Phase 1: Core Implementation
1. Create Gemini provider service
2. Add Gemini endpoint handler
3. Update provider factory

### Phase 2: Testing & Validation
1. Add comprehensive tests
2. Validate with real Gemini API
3. Performance testing

### Phase 3: Documentation
1. Update API documentation
2. Update CLAUDE.md files
3. Add usage examples

## Benefits

1. **Native Format Support**: Direct access to Gemini's native response structure
2. **Backward Compatibility**: Existing OpenAI clients continue working
3. **Performance**: No unnecessary format conversion for Gemini-native clients
4. **Flexibility**: Clients can choose the response format they prefer
5. **Extensibility**: Pattern can be extended to other providers (Anthropic, etc.)

## Future Considerations

### Anthropic Native Endpoint
- Similar pattern: `/v1/messages` (Claude API format)
- Native response format for Anthropic clients

### Provider-Specific Features
- Each endpoint can expose provider-specific features
- No need to maintain compatibility across different response formats

### Client SDKs
- OpenAI SDK clients use `/v1/chat/completions`
- Gemini SDK clients use `/v1/models/{model}:generateContent`
- Native provider SDKs get full feature access