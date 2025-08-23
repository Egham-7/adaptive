package models

// ModelCapability represents a model with its capabilities and constraints
type ModelCapability struct {
	Description             *string  `json:"description,omitempty"`
	Provider                string   `json:"provider"`
	ModelName               string   `json:"model_name"`
	CostPer1MInputTokens    float64  `json:"cost_per_1m_input_tokens"`
	CostPer1MOutputTokens   float64  `json:"cost_per_1m_output_tokens"`
	MaxContextTokens        int      `json:"max_context_tokens"`
	MaxOutputTokens         *int     `json:"max_output_tokens,omitempty"`
	SupportsFunctionCalling bool     `json:"supports_function_calling"`
	LanguagesSupported      []string `json:"languages_supported,omitempty"`
	ModelSizeParams         *string  `json:"model_size_params,omitempty"`
	LatencyTier             *string  `json:"latency_tier,omitempty"`
	TaskType                *string  `json:"task_type,omitempty"`
	Complexity              *string  `json:"complexity,omitempty"`
}
