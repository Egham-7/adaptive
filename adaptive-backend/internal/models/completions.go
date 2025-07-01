package models

import (
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// SelectModelRequest represents a request to select a model for a given prompt and provider.
type SelectModelRequest struct {
	Prompt   string `json:"prompt"`
	Provider string `json:"provider,omitempty"`
}

// ChatCompletionRequest represents a request for a chat completion, including all OpenAI parameters and extensions.
type ChatCompletionRequest struct {
	// Messages comprising the conversation so far.
	Messages []openai.ChatCompletionMessageParamUnion `json:"messages,omitzero"`
	// Model ID used to generate the response.
	Model shared.ChatModel `json:"model,omitzero"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on their
	// existing frequency in the text so far, decreasing the model's likelihood to
	// repeat the same line verbatim.
	FrequencyPenalty param.Opt[float64] `json:"frequency_penalty,omitzero"`
	// Whether to return log probabilities of the output tokens or not. If true,
	// returns the log probabilities of each output token returned in the `content` of
	// `message`.
	Logprobs param.Opt[bool] `json:"logprobs,omitzero"`
	// An upper bound for the number of tokens that can be generated for a completion,
	// including visible output tokens and
	// [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).
	MaxCompletionTokens param.Opt[int64] `json:"max_completion_tokens,omitzero"`
	// The maximum number of [tokens](/tokenizer) that can be generated in the chat
	// completion. This value can be used to control
	// [costs](https://openai.com/api/pricing/) for text generated via API.
	//
	// This value is now deprecated in favor of `max_completion_tokens`, and is not
	// compatible with
	// [o-series models](https://platform.openai.com/docs/guides/reasoning).
	MaxTokens param.Opt[int64] `json:"max_tokens,omitzero"`
	// How many chat completion choices to generate for each input message. Note that
	// you will be charged based on the number of generated tokens across all of the
	// choices. Keep `n` as `1` to minimize costs.
	N param.Opt[int64] `json:"n,omitzero"`
	// Number between -2.0 and 2.0. Positive values penalize new tokens based on
	// whether they appear in the text so far, increasing the model's likelihood to
	// talk about new topics.
	PresencePenalty param.Opt[float64] `json:"presence_penalty,omitzero"`
	// This feature is in Beta. If specified, our system will make a best effort to
	// sample deterministically, such that repeated requests with the same `seed` and
	// parameters should return the same result. Determinism is not guaranteed, and you
	// should refer to the `system_fingerprint` response parameter to monitor changes
	// in the backend.
	Seed param.Opt[int64] `json:"seed,omitzero"`
	// Whether or not to store the output of this chat completion request for use in
	// our [model distillation](https://platform.openai.com/docs/guides/distillation)
	// or [evals](https://platform.openai.com/docs/guides/evals) products.
	Store param.Opt[bool] `json:"store,omitzero"`
	// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
	// make the output more random, while lower values like 0.2 will make it more
	// focused and deterministic. We generally recommend altering this or `top_p` but
	// not both.
	Temperature param.Opt[float64] `json:"temperature,omitzero"`
	// An integer between 0 and 20 specifying the number of most likely tokens to
	// return at each token position, each with an associated log probability.
	// `logprobs` must be set to `true` if this parameter is used.
	TopLogprobs param.Opt[int64] `json:"top_logprobs,omitzero"`
	// An alternative to sampling with temperature, called nucleus sampling, where the
	// model considers the results of the tokens with top_p probability mass. So 0.1
	// means only the tokens comprising the top 10% probability mass are considered.
	//
	// We generally recommend altering this or `temperature` but not both.
	TopP param.Opt[float64] `json:"top_p,omitzero"`
	// Whether to enable
	// [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
	// during tool use.
	ParallelToolCalls param.Opt[bool] `json:"parallel_tool_calls,omitzero"`
	// A stable identifier for your end-users. Used to boost cache hit rates by better
	// bucketing similar requests and to help OpenAI detect and prevent abuse.
	// [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).
	User param.Opt[string] `json:"user,omitzero"`
	// Parameters for audio output. Required when audio output is requested with
	// `modalities: ["audio"]`.
	// [Learn more](https://platform.openai.com/docs/guides/audio).
	Audio openai.ChatCompletionAudioParam `json:"audio,omitzero"`
	// Modify the likelihood of specified tokens appearing in the completion.
	//
	// Accepts a JSON object that maps tokens (specified by their token ID in the
	// tokenizer) to an associated bias value from -100 to 100. Mathematically, the
	// bias is added to the logits generated by the model prior to sampling. The exact
	// effect will vary per model, but values between -1 and 1 should decrease or
	// increase likelihood of selection; values like -100 or 100 should result in a ban
	// or exclusive selection of the relevant token.
	LogitBias map[string]int64 `json:"logit_bias,omitzero"`
	// Set of 16 key-value pairs that can be attached to an object. This can be useful
	// for storing additional information about the object in a structured format, and
	// querying for objects via API or the dashboard.
	//
	// Keys are strings with a maximum length of 64 characters. Values are strings with
	// a maximum length of 512 characters.
	Metadata shared.Metadata `json:"metadata,omitzero"`
	// Output types that you would like the model to generate. Most models are capable
	// of generating text, which is the default:
	//
	// `["text"]`
	//
	// The `gpt-4o-audio-preview` model can also be used to
	// [generate audio](https://platform.openai.com/docs/guides/audio). To request that
	// this model generate both text and audio responses, you can use:
	//
	// `["text", "audio"]`
	//
	// Any of "text", "audio".
	Modalities []string `json:"modalities,omitzero"`
	// **o-series models only**
	//
	// Constrains effort on reasoning for
	// [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
	// supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
	// result in faster responses and fewer tokens used on reasoning in a response.
	//
	// Any of "low", "medium", "high".
	ReasoningEffort shared.ReasoningEffort `json:"reasoning_effort,omitzero"`
	// Specifies the latency tier to use for processing the request. This parameter is
	// relevant for customers subscribed to the scale tier service:
	//
	//   - If set to 'auto', and the Project is Scale tier enabled, the system will
	//     utilize scale tier credits until they are exhausted.
	//   - If set to 'auto', and the Project is not Scale tier enabled, the request will
	//     be processed using the default service tier with a lower uptime SLA and no
	//     latency guarentee.
	//   - If set to 'default', the request will be processed using the default service
	//     tier with a lower uptime SLA and no latency guarentee.
	//   - If set to 'flex', the request will be processed with the Flex Processing
	//     service tier.
	//     [Learn more](https://platform.openai.com/docs/guides/flex-processing).
	//   - When not set, the default behavior is 'auto'.
	//
	// When this parameter is set, the response body will include the `service_tier`
	// utilized.
	//
	// Any of "auto", "default", "flex".
	ServiceTier openai.ChatCompletionNewParamsServiceTier `json:"service_tier,omitzero"`
	// Not supported with latest reasoning models `o3` and `o4-mini`.
	//
	// Up to 4 sequences where the API will stop generating further tokens. The
	// returned text will not contain the stop sequence.
	Stop openai.ChatCompletionNewParamsStopUnion `json:"stop,omitzero"`
	// Options for streaming response. Only set this when you set `stream: true`.
	StreamOptions openai.ChatCompletionStreamOptionsParam `json:"stream_options,omitzero"`

	// Deprecated in favor of `tool_choice`.
	//
	// Controls which (if any) function is called by the model.
	//
	// `none` means the model will not call a function and instead generates a message.
	//
	// `auto` means the model can pick between generating a message or calling a
	// function.
	//
	// Specifying a particular function via `{"name": "my_function"}` forces the model
	// to call that function.
	//
	// `none` is the default when no functions are present. `auto` is the default if
	// functions are present.
	FunctionCall openai.ChatCompletionNewParamsFunctionCallUnion `json:"function_call,omitzero"`
	// Deprecated in favor of `tools`.
	// Static predicted output content, such as the content of a text file that is
	// being regenerated.
	Prediction openai.ChatCompletionPredictionContentParam `json:"prediction,omitzero"`
	// An object specifying the format that the model must output.
	//
	// Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
	// Outputs which ensures the model will match your supplied JSON schema. Learn more
	// in the
	// [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
	//
	// Setting to `{ "type": "json_object" }` enables the older JSON mode, which
	// ensures the message the model generates is valid JSON. Using `json_schema` is
	// preferred for models that support it.
	ResponseFormat openai.ChatCompletionNewParamsResponseFormatUnion `json:"response_format,omitzero"`
	// Controls which (if any) tool is called by the model. `none` means the model will
	// not call any tool and instead generates a message. `auto` means the model can
	// pick between generating a message or calling one or more tools. `required` means
	// the model must call one or more tools. Specifying a particular tool via
	// `{"type": "function", "function": {"name": "my_function"}}` forces the model to
	// call that tool.
	//
	// `none` is the default when no tools are present. `auto` is the default if tools
	// are present.
	ToolChoice openai.ChatCompletionToolChoiceOptionUnionParam `json:"tool_choice,omitzero"`
	// A list of tools the model may call. Currently, only functions are supported as a
	// tool. Use this to provide a list of functions the model may generate JSON inputs
	// for. A max of 128 functions are supported.
	Tools []openai.ChatCompletionToolParam `json:"tools,omitzero"`
	// This tool searches the web for relevant results to use in a response. Learn more
	// about the
	// [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).
	WebSearchOptions openai.ChatCompletionNewParamsWebSearchOptions `json:"web_search_options,omitzero"`

	Stream bool `json:"stream,omitzero"` // Whether to stream the response or not

	ProviderConstraint []string           `json:"provider_constraint,omitempty"`
	CostBias           float32            `json:"cost_bias,omitempty"` // Bias towards cheaper providers
	ComparisonProvider ComparisonProvider `json:"comparison_provider"` // Provider to compare costs against
}

// ToOpenAIParams converts a ChatCompletionRequest to OpenAI's ChatCompletionNewParams.
func (r *ChatCompletionRequest) ToOpenAIParams() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Messages:            r.Messages,
		Model:               r.Model,
		FrequencyPenalty:    r.FrequencyPenalty,
		Logprobs:            r.Logprobs,
		MaxCompletionTokens: r.MaxCompletionTokens,
		MaxTokens:           r.MaxTokens,
		N:                   r.N,
		PresencePenalty:     r.PresencePenalty,
		Seed:                r.Seed,
		Store:               r.Store,
		Temperature:         r.Temperature,
		TopLogprobs:         r.TopLogprobs,
		TopP:                r.TopP,
		ParallelToolCalls:   r.ParallelToolCalls,
		User:                r.User,
		Audio:               r.Audio,
		LogitBias:           r.LogitBias,
		Metadata:            r.Metadata,
		Modalities:          r.Modalities,
		ReasoningEffort:     r.ReasoningEffort,
		ServiceTier:         r.ServiceTier,
		Stop:                r.Stop,
		StreamOptions:       r.StreamOptions,
		FunctionCall:        r.FunctionCall,
		Prediction:          r.Prediction,
		ResponseFormat:      r.ResponseFormat,
		ToolChoice:          r.ToolChoice,
		Tools:               r.Tools,
		WebSearchOptions:    r.WebSearchOptions,
	}
}

// ComparisonProvider represents a provider and model for cost comparison
type ComparisonProvider struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// CompletionUsage extends OpenAI's CompletionUsage with cost savings
type CompletionUsage struct {
	CostSaved float32 `json:"cost_saved,omitempty"`

	// Number of tokens in the generated completion.
	CompletionTokens int64 `json:"completion_tokens"`
	// Number of tokens in the prompt.
	PromptTokens int64 `json:"prompt_tokens"`
	// Total number of tokens used in the request (prompt + completion).
	TotalTokens int64 `json:"total_tokens"`
	// Breakdown of tokens used in a completion.
	CompletionTokensDetails openai.CompletionUsageCompletionTokensDetails `json:"completion_tokens_details"`
	// Breakdown of tokens used in the prompt.
	PromptTokensDetails openai.CompletionUsagePromptTokensDetails `json:"prompt_tokens_details"`
}

// ChatCompletion extends OpenAI's ChatCompletion with enhanced usage
type ChatCompletion struct {
	ID                string                           `json:"id"`
	Choices           []openai.ChatCompletionChoice    `json:"choices"`
	Created           int64                            `json:"created"`
	Model             string                           `json:"model"`
	Object            string                           `json:"object"`
	ServiceTier       openai.ChatCompletionServiceTier `json:"service_tier,omitempty"`
	SystemFingerprint string                           `json:"system_fingerprint,omitempty"`
	Usage             CompletionUsage                  `json:"usage"`
	Providers         []string                         `json:"providers,omitempty"`
}

// ChatCompletionChunkChoiceDelta represents the delta of a chat completion chunk choice with proper JSON handling
type ChatCompletionChunkChoiceDelta struct {
	Content   string                                          `json:"content,omitempty"`
	Refusal   string                                          `json:"refusal,omitempty"`
	Role      string                                          `json:"role,omitempty"`
	ToolCalls []openai.ChatCompletionChunkChoiceDeltaToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionChunkChoice represents a chat completion chunk choice with proper JSON handling
type ChatCompletionChunkChoice struct {
	Delta        ChatCompletionChunkChoiceDelta           `json:"delta"`
	FinishReason string                                   `json:"finish_reason,omitempty"`
	Index        int64                                    `json:"index"`
	Logprobs     openai.ChatCompletionChunkChoiceLogprobs `json:"logprobs"`
}

// ChatCompletionChunk extends OpenAI's ChatCompletionChunk with enhanced usage
type ChatCompletionChunk struct {
	ID                string                                `json:"id"`
	Choices           []ChatCompletionChunkChoice           `json:"choices"`
	Created           int64                                 `json:"created"`
	Model             string                                `json:"model"`
	Object            string                                `json:"object"`
	ServiceTier       openai.ChatCompletionChunkServiceTier `json:"service_tier,omitempty"`
	SystemFingerprint string                                `json:"system_fingerprint,omitempty"`
	Usage             *CompletionUsage                      `json:"usage,omitempty"`
	Providers         []string                              `json:"providers,omitempty"`
}

// ConvertToAdaptive converts OpenAI ChatCompletion to our ChatCompletion
func ConvertToAdaptive(completion *openai.ChatCompletion, costSaved float32, providers []string) *ChatCompletion {
	return &ChatCompletion{
		ID:                completion.ID,
		Choices:           completion.Choices,
		Created:           completion.Created,
		Model:             completion.Model,
		Object:            string(completion.Object),
		ServiceTier:       completion.ServiceTier,
		SystemFingerprint: completion.SystemFingerprint,
		Usage: CompletionUsage{
			CompletionTokens: completion.Usage.CompletionTokens,
			PromptTokens:     completion.Usage.PromptTokens,
			TotalTokens:      completion.Usage.TotalTokens,
			CostSaved:        costSaved,
		},
		Providers: providers,
	}
}

// ConvertChunkToAdaptive converts OpenAI ChatCompletionChunk to our ChatCompletionChunk
func ConvertChunkToAdaptive(chunk *openai.ChatCompletionChunk, costSaved float32, providers []string) *ChatCompletionChunk {
	// Convert choices with proper role handling
	adaptiveChoices := make([]ChatCompletionChunkChoice, len(chunk.Choices))
	for i, choice := range chunk.Choices {
		adaptiveChoices[i] = ChatCompletionChunkChoice{
			Delta: ChatCompletionChunkChoiceDelta{
				Content:   choice.Delta.Content,
				Refusal:   choice.Delta.Refusal,
				Role:      choice.Delta.Role, // This will be omitted if empty due to omitempty
				ToolCalls: choice.Delta.ToolCalls,
			},
			FinishReason: choice.FinishReason,
			Index:        choice.Index,
			Logprobs:     choice.Logprobs,
		}
	}

	adaptive := &ChatCompletionChunk{
		ID:                chunk.ID,
		Choices:           adaptiveChoices,
		Created:           chunk.Created,
		Model:             chunk.Model,
		Object:            string(chunk.Object),
		ServiceTier:       chunk.ServiceTier,
		SystemFingerprint: chunk.SystemFingerprint,
		Providers:         providers,
	}

	// Only add usage if it exists in the original chunk
	if chunk.Usage.CompletionTokens > 0 || chunk.Usage.PromptTokens > 0 || chunk.Usage.TotalTokens > 0 {
		adaptive.Usage = &CompletionUsage{
			CompletionTokens: chunk.Usage.CompletionTokens,
			PromptTokens:     chunk.Usage.PromptTokens,
			TotalTokens:      chunk.Usage.TotalTokens,
			CostSaved:        costSaved,
		}
	}

	return adaptive
}
