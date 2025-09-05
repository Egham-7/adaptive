# PR Plan: Fix Missing `omitzero` Tags in OpenAI Go SDK

## Overview

Fix empty/zero-value field serialization in streaming and non-streaming responses by implementing proper field cleaning in the adaptive-backend format adapter, which will serve as a temporary solution until the upstream OpenAI Go SDK is fixed.

## Problem Analysis

### Root Cause
The OpenAI Go SDK response structs are missing `omitzero` tags on optional fields, causing empty objects and fields to be serialized even when using Go 1.24+ with native `omitzero` support.

### Current State
From analyzing the codebase:

1. **adaptive-backend Go version**: `go 1.24.2` ✅ (supports native `omitzero`)
2. **OpenAI Go SDK version**: `v1.12.0` ❌ (missing `omitzero` tags on response types)
3. **Request structs**: Already have proper `omitzero` tags ✅
4. **Response structs**: Missing `omitzero` tags on optional fields ❌

### Missing `omitzero` Tags in OpenAI Go SDK

#### Response Types Missing Tags:
```go
// ChatCompletion
ServiceTier       ChatCompletionServiceTier `json:"service_tier,nullable"`     // Missing omitzero
SystemFingerprint string                    `json:"system_fingerprint"`        // Missing omitzero
Usage             CompletionUsage           `json:"usage"`                     // Missing omitzero

// ChatCompletionChunk  
ServiceTier       ChatCompletionChunkServiceTier `json:"service_tier,nullable"`  // Missing omitzero
SystemFingerprint string                         `json:"system_fingerprint"`     // Missing omitzero
Usage             CompletionUsage                `json:"usage,nullable"`         // Missing omitzero

// ChatCompletionChoice
Logprobs          ChatCompletionChoiceLogprobs `json:"logprobs,required"`       // Should be nullable,omitzero

// ChatCompletionMessage
Annotations       []ChatCompletionMessageAnnotation   `json:"annotations"`      // Missing omitzero
Audio             ChatCompletionAudio                 `json:"audio,nullable"`   // Missing omitzero
FunctionCall      ChatCompletionMessageFunctionCall  `json:"function_call"`    // Missing omitzero
ToolCalls         []ChatCompletionMessageToolCallUnion `json:"tool_calls"`     // Missing omitzero

// ChatCompletionChunkChoiceDelta
Content           string                                   `json:"content,nullable"`    // Missing omitzero
FunctionCall      ChatCompletionChunkChoiceDeltaFunctionCall `json:"function_call"`     // Missing omitzero
Refusal           string                                   `json:"refusal,nullable"`   // Missing omitzero
Role              string                                   `json:"role"`               // Missing omitzero
ToolCalls         []ChatCompletionChunkChoiceDeltaToolCall `json:"tool_calls"`         // Missing omitzero
```

## Solution Strategy

### Phase 1: Immediate Fix in adaptive-backend

Instead of waiting for upstream OpenAI Go SDK fixes, implement response cleaning in our format adapter:

**File**: `/adaptive-backend/internal/services/format_adapter/openai_to_adaptive.go`

**Location**: Lines 56-98 in `ConvertResponse()` and `ConvertStreamingChunk()` methods

### Implementation Plan

#### 1. Create Response Cleaning Functions

Add utility functions to clean empty fields from OpenAI responses before converting to our format:

```go
// cleanChatCompletion removes empty/zero-value fields from ChatCompletion
func cleanChatCompletion(resp *openai.ChatCompletion) *openai.ChatCompletion {
    cleaned := *resp
    
    // Clean optional fields if they are zero values
    if string(cleaned.ServiceTier) == "" {
        // Set to nil equivalent or remove from JSON serialization
    }
    if cleaned.SystemFingerprint == "" {
        // Handle empty system fingerprint
    }
    if isUsageEmpty(cleaned.Usage) {
        // Handle empty usage
    }
    
    // Clean choices
    cleanedChoices := make([]openai.ChatCompletionChoice, len(cleaned.Choices))
    for i, choice := range cleaned.Choices {
        cleanedChoices[i] = cleanChoice(choice)
    }
    cleaned.Choices = cleanedChoices
    
    return &cleaned
}

// cleanChatCompletionChunk removes empty/zero-value fields from ChatCompletionChunk
func cleanChatCompletionChunk(chunk *openai.ChatCompletionChunk) *openai.ChatCompletionChunk {
    cleaned := *chunk
    
    // Clean optional fields if they are zero values
    if string(cleaned.ServiceTier) == "" {
        // Handle empty service tier
    }
    if cleaned.SystemFingerprint == "" {
        // Handle empty system fingerprint  
    }
    if isUsageEmpty(cleaned.Usage) {
        // Handle empty usage
    }
    
    // Clean choices and deltas
    cleanedChoices := make([]openai.ChatCompletionChunkChoice, len(cleaned.Choices))
    for i, choice := range cleaned.Choices {
        cleanedChoices[i] = cleanChunkChoice(choice)
    }
    cleaned.Choices = cleanedChoices
    
    return &cleaned
}

// Helper functions
func cleanChoice(choice openai.ChatCompletionChoice) openai.ChatCompletionChoice {
    cleaned := choice
    
    // Clean logprobs if empty
    if isLogprobsEmpty(cleaned.Logprobs) {
        // Handle empty logprobs
    }
    
    // Clean message
    cleaned.Message = cleanMessage(cleaned.Message)
    
    return cleaned
}

func cleanMessage(msg openai.ChatCompletionMessage) openai.ChatCompletionMessage {
    cleaned := msg
    
    // Clean optional fields
    if len(cleaned.Annotations) == 0 {
        cleaned.Annotations = nil
    }
    if isAudioEmpty(cleaned.Audio) {
        // Handle empty audio
    }
    if isFunctionCallEmpty(cleaned.FunctionCall) {
        // Handle empty function call
    }
    if len(cleaned.ToolCalls) == 0 {
        cleaned.ToolCalls = nil
    }
    
    return cleaned
}

func cleanChunkChoice(choice openai.ChatCompletionChunkChoice) openai.ChatCompletionChunkChoice {
    cleaned := choice
    
    // Clean logprobs if empty
    if isChunkLogprobsEmpty(cleaned.Logprobs) {
        // Handle empty logprobs
    }
    
    // Clean delta
    cleaned.Delta = cleanDelta(cleaned.Delta)
    
    return cleaned
}

func cleanDelta(delta openai.ChatCompletionChunkChoiceDelta) openai.ChatCompletionChunkChoiceDelta {
    cleaned := delta
    
    // Clean optional fields in delta
    if cleaned.Content == "" {
        // Handle empty content - but be careful, empty string might be intentional in streaming
    }
    if cleaned.Refusal == "" {
        // Handle empty refusal
    }
    if cleaned.Role == "" {
        // Handle empty role
    }
    if isFunctionCallEmpty(cleaned.FunctionCall) {
        // Handle empty function call
    }
    if len(cleaned.ToolCalls) == 0 {
        cleaned.ToolCalls = nil
    }
    
    return cleaned
}

// Utility functions to check if fields are empty
func isUsageEmpty(usage openai.CompletionUsage) bool {
    return usage.PromptTokens == 0 && usage.CompletionTokens == 0 && usage.TotalTokens == 0
}

func isLogprobsEmpty(logprobs openai.ChatCompletionChoiceLogprobs) bool {
    return len(logprobs.Content) == 0 && len(logprobs.Refusal) == 0
}

func isChunkLogprobsEmpty(logprobs openai.ChatCompletionChunkChoiceLogprobs) bool {
    return len(logprobs.Content) == 0 && len(logprobs.Refusal) == 0
}

func isAudioEmpty(audio openai.ChatCompletionAudio) bool {
    // Check if audio object is empty - need to examine the struct
    return false // Placeholder
}

func isFunctionCallEmpty(fc openai.ChatCompletionMessageFunctionCall) bool {
    return fc.Name == "" && fc.Arguments == ""
}
```

#### 2. Update Conversion Methods

Modify the existing methods to use the cleaning functions:

```go
// ConvertResponse - Updated (line 56)
func (c *OpenAIToAdaptiveConverter) ConvertResponse(resp *openai.ChatCompletion, provider string) (*models.ChatCompletion, error) {
    if resp == nil {
        return nil, fmt.Errorf("openai chat completion cannot be nil")
    }

    // Clean the response to remove empty fields
    cleanedResp := cleanChatCompletion(resp)

    return &models.ChatCompletion{
        ID:                cleanedResp.ID,
        Choices:           cleanedResp.Choices,
        Created:           cleanedResp.Created,
        Model:             cleanedResp.Model,
        Object:            string(cleanedResp.Object),
        ServiceTier:       cleanedResp.ServiceTier,
        SystemFingerprint: cleanedResp.SystemFingerprint,
        Usage:             c.convertUsage(cleanedResp.Usage),
        Provider:          provider,
    }, nil
}

// ConvertStreamingChunk - Updated (line 75)  
func (c *OpenAIToAdaptiveConverter) ConvertStreamingChunk(chunk *openai.ChatCompletionChunk, provider string) (*models.ChatCompletionChunk, error) {
    if chunk == nil {
        return nil, fmt.Errorf("openai chat completion chunk cannot be nil")
    }

    // Clean the chunk to remove empty fields
    cleanedChunk := cleanChatCompletionChunk(chunk)

    var usage *models.AdaptiveUsage
    // Check if usage is provided (only in the last chunk typically)
    if cleanedChunk.Usage.CompletionTokens != 0 || cleanedChunk.Usage.PromptTokens != 0 || cleanedChunk.Usage.TotalTokens != 0 {
        converted := c.convertUsage(cleanedChunk.Usage)
        usage = &converted
    }

    return &models.ChatCompletionChunk{
        ID:                cleanedChunk.ID,
        Choices:           cleanedChunk.Choices,
        Created:           cleanedChunk.Created,
        Model:             cleanedChunk.Model,
        Object:            string(cleanedChunk.Object),
        ServiceTier:       cleanedChunk.ServiceTier,
        SystemFingerprint: cleanedChunk.SystemFingerprint,
        Usage:             usage,
        Provider:          provider,
    }, nil
}
```

### Phase 2: Testing Strategy

#### 1. Unit Tests
Create comprehensive tests in `openai_to_adaptive_test.go`:

```go
func TestConvertResponseWithEmptyFields(t *testing.T) {
    converter := NewOpenAIToAdaptiveConverter()
    
    // Test response with empty optional fields
    resp := &openai.ChatCompletion{
        ID:      "test-123",
        Choices: []openai.ChatCompletionChoice{
            {
                FinishReason: "stop",
                Index:        0,
                Message: openai.ChatCompletionMessage{
                    Content: "Hello!",
                    Role:    "assistant",
                    // Leave optional fields empty
                },
            },
        },
        Created: 1234567890,
        Model:   "gpt-4",
        Object:  "chat.completion",
        // Leave ServiceTier, SystemFingerprint, Usage as zero values
    }
    
    adaptiveResp, err := converter.ConvertResponse(resp, "openai")
    require.NoError(t, err)
    
    // Serialize to JSON and verify empty fields are omitted
    data, err := json.Marshal(adaptiveResp)
    require.NoError(t, err)
    
    jsonStr := string(data)
    
    // Should NOT contain zero-value fields
    assert.NotContains(t, jsonStr, `"service_tier"`)
    assert.NotContains(t, jsonStr, `"system_fingerprint"`)
    assert.NotContains(t, jsonStr, `"annotations"`)
    assert.NotContains(t, jsonStr, `"function_call"`)
    assert.NotContains(t, jsonStr, `"tool_calls"`)
}

func TestConvertStreamingChunkWithEmptyFields(t *testing.T) {
    // Similar test for streaming chunks
}
```

#### 2. Integration Tests
Test with actual API responses to ensure the cleaning works end-to-end.

### Phase 3: Long-term Solution

#### Submit PR to OpenAI Go SDK
Once the immediate fix is implemented and tested, create a comprehensive PR to the upstream OpenAI Go SDK with all the missing `omitzero` tags as detailed in the analysis above.

## Benefits of This Approach

1. **Immediate Relief**: Fixes the problem in our codebase without waiting for upstream changes
2. **Centralized**: All cleaning logic is in one place (format adapter)
3. **Backwards Compatible**: No breaking changes to our API consumers
4. **Future-Proof**: When upstream SDK is fixed, we can easily remove our cleaning logic
5. **Testable**: Easy to unit test the cleaning logic

## Implementation Timeline

1. **Week 1**: Implement cleaning functions in format adapter
2. **Week 1**: Add comprehensive unit tests
3. **Week 1**: Integration testing with real API responses
4. **Week 2**: Submit upstream PR to OpenAI Go SDK
5. **Future**: Remove cleaning logic once upstream PR is merged and released

## Files to Modify

### Primary Changes
- `/adaptive-backend/internal/services/format_adapter/openai_to_adaptive.go`

### Testing
- `/adaptive-backend/internal/services/format_adapter/openai_to_adaptive_test.go` (new file)

### Documentation
- Update this plan with implementation details
- Document the temporary nature of the solution

## Success Criteria

- [ ] Empty fields are properly omitted from JSON responses
- [ ] Streaming responses don't include empty deltas
- [ ] No breaking changes to existing API consumers
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance impact is minimal (<1ms per response)
- [ ] Solution is easily removable when upstream is fixed