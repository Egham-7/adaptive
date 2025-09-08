package handlers

import (
	"context"
	"io"
	"time"

	"adaptive-backend/internal/stream/contracts"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

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

	fiberlog.Infof("[%s] Starting stream orchestration for provider: %s", s.requestID, s.processor.Provider())

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

	// Buffer for reading chunks - increased size to handle large JSON frames
	buffer := make([]byte, 32768) // 32KB buffer to accommodate complete JSON frames

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
			return contracts.NewProviderError(s.requestID, s.processor.Provider(), err)
		}

		// Skip empty reads
		if n == 0 {
			continue
		}

		// Process the chunk data
		processedData, err := s.processor.Process(ctx, buffer[:n])
		if err != nil {
			if err == context.Canceled {
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
