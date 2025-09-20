package models

// ModelCapability represents a model with its capabilities and constraints
type ModelCapability struct {
	Description           string   `json:"description,omitzero"`
	Provider              string   `json:"provider,omitzero"`
	ModelName             string   `json:"model_name,omitzero"`
	CostPer1MInputTokens  float64  `json:"cost_per_1m_input_tokens,omitzero"`
	CostPer1MOutputTokens float64  `json:"cost_per_1m_output_tokens,omitzero"`
	MaxContextTokens      int      `json:"max_context_tokens,omitzero"`
	MaxOutputTokens       int      `json:"max_output_tokens,omitzero"`
	SupportsToolCalling   bool     `json:"supports_tool_calling,omitzero"`
	LanguagesSupported    []string `json:"languages_supported,omitzero"`
	ModelSizeParams       string   `json:"model_size_params,omitzero"`
	LatencyTier           string   `json:"latency_tier,omitzero"`
	TaskType              string   `json:"task_type,omitzero"`
	Complexity            string   `json:"complexity,omitzero"`
}
