package handlers

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"

	"adaptive-backend/internal/services/stream/contracts"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// BufferConfig holds configuration for streaming buffers
type BufferConfig struct {
	DefaultSize   int
	MaxSize       int
	ProviderSizes map[string]int // Provider-specific buffer sizes
}

// DefaultBufferConfig returns default buffer configuration
func DefaultBufferConfig() *BufferConfig {
	return &BufferConfig{
		DefaultSize: 32768, // 32KB default
		MaxSize:     65536, // 64KB maximum
		ProviderSizes: map[string]int{
			"openai":    16384, // 16KB for OpenAI (smaller chunks)
			"anthropic": 32768, // 32KB for Anthropic (larger chunks)
			"google":    24576, // 24KB for Google AI
			"deepseek":  20480, // 20KB for DeepSeek
			"groq":      8192,  // 8KB for Groq (fast, smaller chunks)
		},
	}
}

// bufferPool manages reusable byte buffers for streaming
var bufferPool = sync.Pool{
	New: func() interface{} {
		buf := make([]byte, DefaultBufferConfig().DefaultSize)
		return &buf
	},
}

// providerBufferPools holds provider-specific buffer pools
var (
	providerBufferPools = make(map[string]*sync.Pool)
	poolInitOnce        sync.Once
)

// initializeProviderPools sets up provider-specific buffer pools
func initializeProviderPools() {
	config := DefaultBufferConfig()
	for provider, size := range config.ProviderSizes {
		providerBufferPools[provider] = &sync.Pool{
			New: func(bufSize int) func() interface{} {
				return func() interface{} {
					buf := make([]byte, bufSize)
					return &buf
				}
			}(size),
		}
	}
}

// getBuffer retrieves a buffer from the appropriate pool based on provider
func getBuffer(provider string) []byte {
	poolInitOnce.Do(initializeProviderPools)

	if pool, exists := providerBufferPools[provider]; exists {
		return *pool.Get().(*[]byte)
	}

	// Fallback to default pool
	return *bufferPool.Get().(*[]byte)
}

// putBuffer returns a buffer to the appropriate pool
func putBuffer(buffer []byte, provider string) {
	// Clear buffer to prevent data leaks
	for i := range buffer {
		buffer[i] = 0
	}

	poolInitOnce.Do(initializeProviderPools)

	if pool, exists := providerBufferPools[provider]; exists {
		// Verify buffer size matches pool expectation
		config := DefaultBufferConfig()
		if expectedSize, ok := config.ProviderSizes[provider]; ok && len(buffer) == expectedSize {
			pool.Put(&buffer)
			return
		}
	}

	// Fallback to default pool if size matches
	if len(buffer) == DefaultBufferConfig().DefaultSize {
		bufferPool.Put(&buffer)
	}
	// If buffer size doesn't match any pool, let GC handle it
}

// StreamOrchestrator coordinates the streaming pipeline
type StreamOrchestrator struct {
	reader    contracts.StreamReader
	processor contracts.ChunkProcessor
	requestID string
}

// NewStreamOrchestrator creates a new stream orchestrator
func NewStreamOrchestrator(reader contracts.StreamReader, processor contracts.ChunkProcessor, requestID string) *StreamOrchestrator {
	return &StreamOrchestrator{
		reader:    reader,
		processor: processor,
		requestID: requestID,
	}
}

// Handle orchestrates the complete streaming pipeline
func (s *StreamOrchestrator) Handle(ctx context.Context, writer contracts.StreamWriter) error {
	startTime := time.Now()
	var totalChunks int64
	var totalBytes int64

	providerName := s.processor.Provider()
	fiberlog.Infof("[%s] Starting stream orchestration for provider: %s", s.requestID, providerName)

	// Get buffer from pool based on provider
	buffer := getBuffer(providerName)
	defer putBuffer(buffer, providerName)

	// Ensure cleanup
	defer func() {
		duration := time.Since(startTime)
		fiberlog.Infof("[%s] Stream completed: %d chunks, %d bytes in %v (%.2f KB/s)",
			s.requestID, totalChunks, totalBytes, duration, float64(totalBytes)/duration.Seconds()/1024)

		// Close resources
		if err := s.reader.Close(); err != nil {
			fiberlog.Errorf("[%s] Error closing reader: %v", s.requestID, err)
		}
		if err := writer.Close(); err != nil && !contracts.IsExpectedError(err) {
			fiberlog.Errorf("[%s] Error closing writer: %v", s.requestID, err)
		}
	}()

	for {
		// Check for context cancellation first
		select {
		case <-ctx.Done():
			fiberlog.Infof("[%s] Context cancelled, stopping stream", s.requestID)
			return contracts.NewClientDisconnectError(s.requestID)
		default:
		}

		// Read chunk from stream
		n, err := s.reader.Read(buffer)
		if err == io.EOF {
			// Natural end of stream
			fiberlog.Infof("[%s] Stream completed naturally", s.requestID)
			return contracts.NewStreamCompleteError(s.requestID)
		}
		if err != nil {
			return contracts.NewProviderError(s.requestID, providerName, err)
		}

		// Skip empty reads
		if n == 0 {
			continue
		}

		// Process the chunk data
		processedData, err := s.processor.Process(ctx, buffer[:n])
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return contracts.NewClientDisconnectError(s.requestID)
			}
			return contracts.NewInternalError(s.requestID, "chunk processing failed", err)
		}

		// Skip empty processed data
		if len(processedData) == 0 {
			continue
		}

		// Write processed data
		if err := writer.Write(processedData); err != nil {
			if contracts.IsClientDisconnect(err) {
				fiberlog.Infof("[%s] Client disconnected during write", s.requestID)
				return err // Return as-is for proper classification
			}
			return contracts.NewInternalError(s.requestID, "write failed", err)
		}

		// Flush data
		if err := writer.Flush(); err != nil {
			if contracts.IsClientDisconnect(err) {
				fiberlog.Infof("[%s] Client disconnected during flush", s.requestID)
				return err // Return as-is for proper classification
			}
			return contracts.NewInternalError(s.requestID, "flush failed", err)
		}

		// Update metrics
		totalChunks++
		totalBytes += int64(len(processedData))

		// Periodic logging for long streams
		if totalChunks%100 == 0 {
			duration := time.Since(startTime)
			throughput := float64(totalBytes) / duration.Seconds() / 1024
			fiberlog.Debugf("[%s] Stream progress: %d chunks, %d bytes, %.2f KB/s",
				s.requestID, totalChunks, totalBytes, throughput)
		}
	}
}

// RequestID returns the request ID
func (s *StreamOrchestrator) RequestID() string {
	return s.requestID
}
