package models

// SelectModelResponse represents the response for model selection
type SelectModelResponse struct {
	// The optimized chat completion request ready for execution
	Request *ChatCompletionRequest `json:"request"`
	// Additional metadata about the selection
	Metadata SelectionMetadata `json:"metadata"`
}

// SelectionMetadata provides additional information about the model selection
type SelectionMetadata struct {
	Provider    string  `json:"provider"`
	Model       string  `json:"model"`
	Reasoning   string  `json:"reasoning,omitempty"`
	CostPer1M   float64 `json:"cost_per_1m_tokens,omitempty"`
	Complexity  string  `json:"complexity,omitempty"`
	CacheSource string  `json:"cache_source,omitempty"`
}
