package pricing

// ModelPricing represents the pricing for input and output tokens per million
type ModelPricing struct {
	InputTokensPerMillion  float32 `json:"input_tokens_per_million"`
	OutputTokensPerMillion float32 `json:"output_tokens_per_million"`
}

// ProviderPricing maps model names to their pricing
type ProviderPricing map[string]ModelPricing

// GlobalPricing contains pricing for all providers
var GlobalPricing = map[string]ProviderPricing{
	"openai": {
		"gpt-3.5-turbo":          {InputTokensPerMillion: 0.50, OutputTokensPerMillion: 1.50},
		"gpt-4o":                 {InputTokensPerMillion: 3.00, OutputTokensPerMillion: 10.00},
		"gpt-4o-mini":           {InputTokensPerMillion: 0.15, OutputTokensPerMillion: 0.60},
		"gpt-4":                 {InputTokensPerMillion: 30.00, OutputTokensPerMillion: 60.00},
		"gpt-4-turbo":           {InputTokensPerMillion: 10.00, OutputTokensPerMillion: 30.00},
	},
	"anthropic": {
		"claude-3.5-sonnet":     {InputTokensPerMillion: 3.00, OutputTokensPerMillion: 15.00},
		"claude-3.5-haiku":      {InputTokensPerMillion: 0.80, OutputTokensPerMillion: 4.00},
		"claude-3-opus":         {InputTokensPerMillion: 15.00, OutputTokensPerMillion: 75.00},
		"claude-4-sonnet":       {InputTokensPerMillion: 3.00, OutputTokensPerMillion: 15.00},
	},
	"google": {
		"gemini-2.5-pro":        {InputTokensPerMillion: 1.25, OutputTokensPerMillion: 10.00}, // up to 200k tokens
		"gemini-2.5-pro-large":  {InputTokensPerMillion: 2.50, OutputTokensPerMillion: 15.00}, // over 200k tokens
		"gemini-1.5-flash":      {InputTokensPerMillion: 1.25, OutputTokensPerMillion: 5.00},
		"gemini-2.0-flash":      {InputTokensPerMillion: 0.10, OutputTokensPerMillion: 0.40},
		"gemini-pro":            {InputTokensPerMillion: 1.25, OutputTokensPerMillion: 5.00},
	},
	"deepseek": {
		"deepseek-chat":         {InputTokensPerMillion: 0.27, OutputTokensPerMillion: 1.10}, // cache miss
		"deepseek-reasoner":     {InputTokensPerMillion: 0.55, OutputTokensPerMillion: 2.19}, // cache miss
	},
	"groq": {
		"llama-4-scout-17b-16e-instruct":    {InputTokensPerMillion: 0.11, OutputTokensPerMillion: 0.34},
		"llama-4-maverick-17b-128e-instruct": {InputTokensPerMillion: 0.20, OutputTokensPerMillion: 0.60},
		"llama-guard-4-12b":                 {InputTokensPerMillion: 0.20, OutputTokensPerMillion: 0.20},
		"deepseek-r1-distill-llama-70b":     {InputTokensPerMillion: 0.75, OutputTokensPerMillion: 0.99},
		"qwen-qwq-32b":                      {InputTokensPerMillion: 0.29, OutputTokensPerMillion: 0.39},
		"mistral-saba-24b":                  {InputTokensPerMillion: 0.79, OutputTokensPerMillion: 0.79},
		"llama-3.3-70b-versatile":          {InputTokensPerMillion: 0.59, OutputTokensPerMillion: 0.79},
		"llama-3.1-8b-instant":             {InputTokensPerMillion: 0.05, OutputTokensPerMillion: 0.08},
		"llama3-70b-8192":                  {InputTokensPerMillion: 0.59, OutputTokensPerMillion: 0.79},
		"llama3-8b-8192":                   {InputTokensPerMillion: 0.05, OutputTokensPerMillion: 0.08},
		"mixtral-8x7b-32768":               {InputTokensPerMillion: 0.27, OutputTokensPerMillion: 0.27},
		"gemma-7b-it":                      {InputTokensPerMillion: 0.10, OutputTokensPerMillion: 0.10},
		"gemma2-9b-it":                     {InputTokensPerMillion: 0.10, OutputTokensPerMillion: 0.10},
	},
	"grok": {
		"grok-3":        {InputTokensPerMillion: 3.00, OutputTokensPerMillion: 15.00},
		"grok-3-mini":   {InputTokensPerMillion: 0.30, OutputTokensPerMillion: 0.50},
		"grok-3-fast":   {InputTokensPerMillion: 5.00, OutputTokensPerMillion: 25.00},
		"grok-beta":     {InputTokensPerMillion: 38.15, OutputTokensPerMillion: 114.44}, // Calculated from beta pricing
	},
}

// CalculateCost calculates the cost for given input and output tokens
func CalculateCost(provider, model string, inputTokens, outputTokens int64) float32 {
	providerPricing, exists := GlobalPricing[provider]
	if !exists {
		return 0.0
	}
	
	modelPricing, exists := providerPricing[model]
	if !exists {
		return 0.0
	}
	
	inputCost := float32(inputTokens) / 1000000.0 * modelPricing.InputTokensPerMillion
	outputCost := float32(outputTokens) / 1000000.0 * modelPricing.OutputTokensPerMillion
	
	return inputCost + outputCost
}

// CalculateCostSaved calculates the cost savings between selected provider and comparison provider
func CalculateCostSaved(selectedProvider, selectedModel string, comparisonProvider, comparisonModel string, inputTokens, outputTokens int64) float32 {
	selectedCost := CalculateCost(selectedProvider, selectedModel, inputTokens, outputTokens)
	comparisonCost := CalculateCost(comparisonProvider, comparisonModel, inputTokens, outputTokens)
	
	return comparisonCost - selectedCost
}

// GetModelPricing returns the pricing for a specific model
func GetModelPricing(provider, model string) (ModelPricing, bool) {
	providerPricing, exists := GlobalPricing[provider]
	if !exists {
		return ModelPricing{}, false
	}
	
	modelPricing, exists := providerPricing[model]
	return modelPricing, exists
}