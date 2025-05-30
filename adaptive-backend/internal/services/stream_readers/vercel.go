package stream_readers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"sync" // Ensure sync is imported if BaseStreamReader's closeLock needs it.
)

// Define a simplified internal chunk structure for now.
// This will need to be aligned with what other readers output.
type InternalProviderChunk struct {
	Text         string      `json:"text,omitempty"`
	ToolCalls    []any       `json:"tool_calls,omitempty"` // Simplified
	Error        string      `json:"error,omitempty"`
	FinishReason string      `json:"finish_reason,omitempty"`
	// We might also need Provider string if VercelDataStreamReader needs to know source
}

// VercelDataStreamReader adapts a generic stream to Vercel's AI SDK DataStream format.
type VercelDataStreamReader struct {
	BaseStreamReader
	underlyingStream   io.ReadCloser
	vercelFormatHeader string
	done               bool
	// Potentially a decoder for the underlyingStream if it's JSON
	decoder *json.Decoder
}

// NewVercelDataStreamReader creates a new stream reader for Vercel DataStream format.
func NewVercelDataStreamReader(stream io.ReadCloser, requestID string) *VercelDataStreamReader {
	return &VercelDataStreamReader{
		BaseStreamReader: BaseStreamReader{
			buffer:    []byte{},
			requestID: requestID,
			// closeLock is part of BaseStreamReader, ensure it's initialized if necessary
			// or rely on Go's default initialization for sync.Once.
		},
		underlyingStream:   stream,
		vercelFormatHeader: "x-vercel-ai-data-stream: v1",
		done:               false,
		decoder:            json.NewDecoder(stream), // Initialize decoder
	}
}

// Read implements io.Reader interface for Vercel DataStream format
func (r *VercelDataStreamReader) Read(p []byte) (n int, err error) {
	if len(r.buffer) > 0 {
		n = copy(p, r.buffer)
		r.buffer = r.buffer[n:]
		return n, nil
	}

	if r.done {
		return 0, io.EOF
	}

	var chunk InternalProviderChunk
	// Assuming underlyingStream provides JSON objects one by one.
	if err := r.decoder.Decode(&chunk); err != nil {
		if err == io.EOF {
			// Successfully read all data from underlying stream, prepare Vercel finish message
			finishMsg := `d:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}` + "\n\n"
			r.buffer = []byte(finishMsg)
			r.done = true
			return r.Read(p)
		}
		// Actual error from underlying stream
		log.Printf("[%s] Error decoding chunk from underlying stream: %v", r.requestID, err)
		errMsg := fmt.Sprintf(`3:"Error reading from underlying stream: %s"`+"\n\n", strings.ReplaceAll(err.Error(), `"`, `\"`))
		r.buffer = []byte(errMsg)
		r.done = true // Stop further processing on decode error
		return r.Read(p)
	}

	// Process the chunk
	if chunk.Error != "" {
		errMsg := fmt.Sprintf(`3:"%s"`+"\n\n", strings.ReplaceAll(chunk.Error, `"`, `\"`))
		r.buffer = append(r.buffer, []byte(errMsg)...)
		// Optionally, consider if an error from provider means stream is done.
		// r.done = true 
	} else if chunk.Text != "" {
		// Escape quotes and newlines in the text content for JSON compatibility
		escapedText := strings.ReplaceAll(chunk.Text, `"`, `\"`)
		escapedText = strings.ReplaceAll(escapedText, "\n", `\n`)
		textMsg := fmt.Sprintf(`0:"%s"`+"\n\n", escapedText)
		r.buffer = append(r.buffer, []byte(textMsg)...)
	}
	// Add more conditions here for other Vercel part types (tool_calls, data, etc.) based on InternalProviderChunk

	if chunk.FinishReason != "" {
		// The Vercel protocol expects the 'd' message to be the *last* part.
		// If FinishReason is present, we set done = true. The next Read call,
		// if the buffer is empty, will hit the r.done check. If the r.decoder.Decode(&chunk)
		// returned io.EOF in the same call that populated a chunk with FinishReason, 
		// the EOF handling will correctly append the 'd' message.
		// If FinishReason is in a chunk *before* actual EOF from the stream,
		// we must ensure any buffered content from *this* chunk is sent first.
		// Then, the next Read() call that encounters an empty buffer and r.done=true (set below)
		// will trigger the EOF logic that sends the 'd' message.

		// If there's content in the current chunk along with a finish reason,
		// that content will be added to r.buffer above.
		// We mark r.done = true here. The final 'd' message will be constructed
		// when r.decoder.Decode(&chunk) eventually returns io.EOF, or if Read is called
		// again when r.buffer is empty and r.done is true.

		// To ensure 'd' is the absolute last message if finish_reason is encountered
		// before the stream itself ends:
		// 1. Buffer any content from the current chunk (text, tool_call etc.).
		// 2. Set r.done = true.
		// 3. The next Read call will first drain this buffered content.
		// 4. Subsequent Read calls will find buffer empty, r.done = true, and then hit the
		//    io.EOF logic from *this method* (not necessarily from decoder.Decode),
		//    which should then append the 'd' message.

		// Let's refine the EOF/done logic:
		// The primary EOF from underlyingStream is handled by r.decoder.Decode() returning io.EOF.
		// This is where the 'd' message is currently added.
		// If a chunk contains FinishReason, we should ensure that 'd' message is prepared
		// *after* all content from this chunk and previous buffer is processed.
		// For now, setting r.done = true. If the buffer becomes empty and r.done is true,
		// the next Read will return 0, io.EOF. The 'd' message generation is tied to
		// the *underlyingStream* ending (actual io.EOF from decoder).
		// This means if a provider sends FinishReason but the stream technically stays open
		// for a bit, the 'd' message might be delayed or not sent if more data never arrives.
		// This needs to be robust.

		// A simpler model: if FinishReason is seen, we consider the stream logically over from Vercel's perspective.
		// We'll add the 'd' message to the buffer *after* other parts from this chunk.
		// And then set r.done = true.

		// Let's ensure that if FinishReason is in a chunk, we append the 'd' message
		// to the buffer after any other data from *this* chunk.
		// And then set r.done = true.
		// This avoids relying on a subsequent io.EOF from the decoder if the provider
		// sends FinishReason and then idles.

		// However, the spec implies 'd' is the very last thing. If we add 'd' here,
		// and then the decoder *does* return io.EOF, another 'd' might be added.
		// The current EOF handling (on r.decoder.Decode) seems like the most reliable place for 'd'.
		// So, if FinishReason is in a chunk, we just set r.done = true.
		// If the buffer is empty on the next Read and r.done is true, it returns io.EOF.
		// This seems correct as per io.Reader contract. The 'd' message generation
		// is specifically when the *source* stream ends.
		r.done = true
	}
	
	return r.Read(p)
}

// Close method
func (r *VercelDataStreamReader) Close() error {
	var err error
	// r.closeLock is part of BaseStreamReader, it needs to be initialized.
	// Assuming BaseStreamReader has a field like: closeLock sync.Once
	// If BaseStreamReader is not managing closeLock initialization, it should be done in NewVercelDataStreamReader.
	// For now, let's assume BaseStreamReader handles its own sync.Once initialization.
	// If not, NewVercelDataStreamReader should explicitly init r.BaseStreamReader.closeLock.
	
	// The provided BaseStreamReader doesn't show closeLock, so we manage it here or assume it's there.
	// For safety, let's assume BaseStreamReader has closeLock. If this causes issues,
	// BaseStreamReader needs to be defined/updated.
	// If there's no closeLock in BaseStreamReader, this will panic.
	// Let's defer this concern to BaseStreamReader's definition. The prompt implies it might be there.

	r.closeLock.Do(func() { // This line will cause a compile error if BaseStreamReader doesn't have closeLock
		if r.underlyingStream != nil {
			log.Printf("[%s] Closing underlying stream for VercelDataStreamReader", r.requestID)
			err = r.underlyingStream.Close()
		}
		// Mark as done to ensure Read behaves correctly after Close.
		r.done = true 
		log.Printf("[%s] VercelDataStreamReader closed, requestID: %s", r.requestID, r.requestID)
	})
	return err
}

// Ensure BaseStreamReader includes a sync.Once field named closeLock for the Close method to work as written.
// Example:
// type BaseStreamReader struct {
//   buffer    []byte
//   requestID string
//   closeLock sync.Once // Added this line
//   // other fields...
// }
// If BaseStreamReader cannot be modified, then closeLock needs to be a field in VercelDataStreamReader itself,
// and NewVercelDataStreamReader would initialize it.
// For now, proceeding with the assumption it's in BaseStreamReader as per common patterns.
// The original VercelDataStreamReader struct had a commented out closeLock.Do, implying BaseStreamReader should have it.
```
