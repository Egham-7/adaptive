package format_adapter

import (
	"adaptive-backend/internal/models"
	"fmt"
	"strings"
)

// SupportedFormat represents the API format types we support
type SupportedFormat string

const (
	FormatOpenAI    SupportedFormat = "openai"
	FormatAnthropic SupportedFormat = "anthropic"
)

// ConversionPair represents a source->target format conversion
type ConversionPair struct {
	Source SupportedFormat
	Target SupportedFormat
}

// FormatAdapter handles all format conversions and detection
type FormatAdapter struct {
	// Registry of conversion functions for requests
	requestConverters map[ConversionPair]func(interface{}) (interface{}, error)
	
	// Registry of conversion functions for responses  
	responseConverters map[ConversionPair]func(interface{}, string) (interface{}, error)
	
	// Registry of conversion functions for streaming chunks
	streamConverters map[ConversionPair]func(interface{}, string) (interface{}, error)
}

// NewFormatAdapter creates a new format adapter with built-in converters
func NewFormatAdapter() *FormatAdapter {
	fa := &FormatAdapter{
		requestConverters:  make(map[ConversionPair]func(interface{}) (interface{}, error)),
		responseConverters: make(map[ConversionPair]func(interface{}, string) (interface{}, error)),
		streamConverters:   make(map[ConversionPair]func(interface{}, string) (interface{}, error)),
	}
	
	// Register built-in converters
	fa.registerBuiltinConverters()
	
	return fa
}

// ConvertRequest converts a request between formats
func (fa *FormatAdapter) ConvertRequest(request interface{}, sourceFormat, targetFormat SupportedFormat) (interface{}, error) {
	if sourceFormat == targetFormat {
		return request, nil
	}
	
	pair := ConversionPair{Source: sourceFormat, Target: targetFormat}
	converter, exists := fa.requestConverters[pair]
	if !exists {
		return nil, fmt.Errorf("no converter available for %s -> %s", sourceFormat, targetFormat)
	}
	
	return converter(request)
}

// ConvertResponse converts a response between formats
func (fa *FormatAdapter) ConvertResponse(response interface{}, sourceFormat, targetFormat SupportedFormat, provider string) (interface{}, error) {
	if sourceFormat == targetFormat {
		return response, nil
	}
	
	pair := ConversionPair{Source: sourceFormat, Target: targetFormat}
	converter, exists := fa.responseConverters[pair]
	if !exists {
		return nil, fmt.Errorf("no response converter available for %s -> %s", sourceFormat, targetFormat)
	}
	
	return converter(response, provider)
}

// ConvertStreamingChunk converts a streaming chunk between formats
func (fa *FormatAdapter) ConvertStreamingChunk(chunk interface{}, sourceFormat, targetFormat SupportedFormat, provider string) (interface{}, error) {
	if sourceFormat == targetFormat {
		return chunk, nil
	}
	
	pair := ConversionPair{Source: sourceFormat, Target: targetFormat}
	converter, exists := fa.streamConverters[pair]
	if !exists {
		return nil, fmt.Errorf("no streaming converter available for %s -> %s", sourceFormat, targetFormat)
	}
	
	return converter(chunk, provider)
}

// DetectProviderFormat determines the native format for a provider
func (fa *FormatAdapter) DetectProviderFormat(provider string, config *models.ProviderConfig, endpointDefault SupportedFormat) SupportedFormat {
	// If provider config specifies native format, use that
	if config != nil && config.NativeFormat != "" {
		format := SupportedFormat(strings.ToLower(config.NativeFormat))
		if fa.isValidFormat(format) {
			return format
		}
	}
	
	// Known provider defaults (can be overridden by config)
	knownFormats := map[string]SupportedFormat{
		"anthropic": FormatAnthropic,
		"claude":    FormatAnthropic,
		"openai":    FormatOpenAI,
		"gemini":    FormatOpenAI,
		"deepseek":  FormatOpenAI,
		"groq":      FormatOpenAI,
	}
	
	if format, exists := knownFormats[strings.ToLower(provider)]; exists {
		return format
	}
	
	// Default to the endpoint's native format
	return endpointDefault
}

// IsConversionNeeded checks if format conversion is required
func (fa *FormatAdapter) IsConversionNeeded(requestFormat, providerFormat SupportedFormat) bool {
	return requestFormat != providerFormat
}

// SupportsConversion checks if conversion between formats is supported
func (fa *FormatAdapter) SupportsConversion(sourceFormat, targetFormat SupportedFormat) bool {
	if sourceFormat == targetFormat {
		return true
	}
	
	pair := ConversionPair{Source: sourceFormat, Target: targetFormat}
	_, hasRequest := fa.requestConverters[pair]
	_, hasResponse := fa.responseConverters[pair]
	
	return hasRequest && hasResponse
}

// GetSupportedFormats returns all supported formats
func (fa *FormatAdapter) GetSupportedFormats() []SupportedFormat {
	return []SupportedFormat{FormatOpenAI, FormatAnthropic}
}

// Helper methods

func (fa *FormatAdapter) isValidFormat(format SupportedFormat) bool {
	supportedFormats := fa.GetSupportedFormats()
	for _, supported := range supportedFormats {
		if format == supported {
			return true
		}
	}
	return false
}

// registerBuiltinConverters registers the built-in OpenAI <-> Anthropic converters
func (fa *FormatAdapter) registerBuiltinConverters() {
	// OpenAI -> Anthropic request conversion
	fa.requestConverters[ConversionPair{FormatOpenAI, FormatAnthropic}] = func(req interface{}) (interface{}, error) {
		openaiReq, ok := req.(*models.ChatCompletionRequest)
		if !ok {
			return nil, fmt.Errorf("expected *models.ChatCompletionRequest, got %T", req)
		}
		
		converter := NewOpenAIToAnthropicConverter()
		return converter.ConvertRequest(openaiReq)
	}
	
	// Anthropic -> OpenAI request conversion  
	fa.requestConverters[ConversionPair{FormatAnthropic, FormatOpenAI}] = func(req interface{}) (interface{}, error) {
		// TODO: Implement proper Anthropic to OpenAI conversion
		return nil, fmt.Errorf("anthropic to openai request conversion not yet implemented")
	}
	
	// OpenAI -> Anthropic response conversion
	fa.responseConverters[ConversionPair{FormatOpenAI, FormatAnthropic}] = func(resp interface{}, provider string) (interface{}, error) {
		openaiResp, ok := resp.(*models.ChatCompletion)
		if !ok {
			return nil, fmt.Errorf("expected *models.ChatCompletion, got %T", resp)
		}
		
		converter := NewOpenAIToAnthropicConverter()
		return converter.ConvertResponse(openaiResp, provider)
	}
	
	// Anthropic -> OpenAI response conversion
	fa.responseConverters[ConversionPair{FormatAnthropic, FormatOpenAI}] = func(resp interface{}, provider string) (interface{}, error) {
		// TODO: Implement proper Anthropic to OpenAI response conversion
		return nil, fmt.Errorf("anthropic to openai response conversion not yet implemented")
	}
	
	// Streaming converters...
	fa.streamConverters[ConversionPair{FormatOpenAI, FormatAnthropic}] = func(chunk interface{}, provider string) (interface{}, error) {
		openaiChunk, ok := chunk.(*models.ChatCompletionChunk)
		if !ok {
			return nil, fmt.Errorf("expected *models.ChatCompletionChunk, got %T", chunk)
		}
		
		converter := NewOpenAIToAnthropicConverter()
		return converter.ConvertStreamingChunk(openaiChunk, provider)
	}
	
	fa.streamConverters[ConversionPair{FormatAnthropic, FormatOpenAI}] = func(chunk interface{}, provider string) (interface{}, error) {
		// TODO: Implement proper Anthropic to OpenAI streaming conversion
		return nil, fmt.Errorf("anthropic to openai streaming conversion not yet implemented")
	}
}