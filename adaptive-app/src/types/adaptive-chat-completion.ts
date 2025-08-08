import { z } from "zod";

// Cache tier constants
export const CACHE_TIER_VALUES = [
	"semantic_exact",
	"semantic_similar",
	"prompt_response",
] as const;

export const cacheTierSchema = z.enum(CACHE_TIER_VALUES);
export type CacheTier = z.infer<typeof cacheTierSchema>;

// Export individual constants for backward compatibility
export const CACHE_TIER_SEMANTIC_EXACT = CACHE_TIER_VALUES[0];
export const CACHE_TIER_SEMANTIC_SIMILAR = CACHE_TIER_VALUES[1];
export const CACHE_TIER_PROMPT_RESPONSE = CACHE_TIER_VALUES[2];

// Model capability schema
export const modelCapabilitySchema = z.object({
	description: z.string().optional(),
	provider: z.string(),
	model_name: z.string(),
	cost_per_1m_input_tokens: z.number(),
	cost_per_1m_output_tokens: z.number(),
	max_context_tokens: z.number(),
	max_output_tokens: z.number().optional(),
	supports_function_calling: z.boolean(),
	languages_supported: z.array(z.string()).optional(),
	model_size_params: z.string().optional(),
	latency_tier: z.string().optional(),
	task_type: z.string().optional(),
	complexity: z.string().optional(),
});

// Fallback mode schema
export const fallbackModeSchema = z.enum(["sequential", "parallel"]);

// Fallback config schema
export const fallbackConfigSchema = z.object({
	enabled: z.boolean().optional(),
	mode: fallbackModeSchema.optional(),
});

// Protocol manager config schema
export const protocolManagerConfigSchema = z.object({
	models: z.array(modelCapabilitySchema).optional(),
	cost_bias: z.number().optional(),
	complexity_threshold: z.number().optional(),
	token_threshold: z.number().optional(),
});

// Cache config schema
export const cacheConfigSchema = z.object({
	enabled: z.boolean(),
	semantic_threshold: z.number().optional(),
});

// Prompt cache config schema
export const promptCacheConfigSchema = z.object({
	enabled: z.boolean().optional(),
	ttl: z.number().optional(),
});

// Chat completion message schema
export const chatCompletionMessageSchema = z.object({
	role: z.enum(["system", "user", "assistant", "tool", "function"]),
	content: z.union([z.string(), z.null(), z.array(z.any())]).optional(),
	name: z.string().optional(),
	tool_calls: z.array(z.any()).optional(),
	tool_call_id: z.string().optional(),
	function_call: z.any().optional(),
});

// Main chat completion request schema
export const adaptiveChatCompletionRequestSchema = z.object({
	// Core OpenAI parameters
	messages: z.array(chatCompletionMessageSchema),
	model: z.string(),
	frequency_penalty: z.number().min(-2).max(2).optional(),
	logprobs: z.boolean().optional(),
	max_completion_tokens: z.number().positive().optional(),
	max_tokens: z.number().positive().optional(),
	n: z.number().positive().optional(),
	presence_penalty: z.number().min(-2).max(2).optional(),
	seed: z.number().optional(),
	store: z.boolean().optional(),
	temperature: z.number().min(0).max(2).optional(),
	top_logprobs: z.number().min(0).max(20).optional(),
	top_p: z.number().min(0).max(1).optional(),
	parallel_tool_calls: z.boolean().optional(),
	user: z.string().optional(),

	// OpenAI advanced parameters
	audio: z.any().optional(),
	logit_bias: z.record(z.string(), z.number()).optional(),
	metadata: z.record(z.string(), z.any()).optional(),
	modalities: z.array(z.string()).optional(),
	reasoning_effort: z.enum(["low", "medium", "high"]).optional(),
	service_tier: z.enum(["auto", "default", "flex"]).optional(),
	stop: z.union([z.string(), z.array(z.string())]).optional(),
	stream_options: z.any().optional(),
	function_call: z.any().optional(),
	prediction: z.any().optional(),
	response_format: z.any().optional(),
	tool_choice: z.any().optional(),
	tools: z.array(z.any()).optional(),
	web_search_options: z.any().optional(),

	// Adaptive-specific parameters
	stream: z.boolean().optional(),
	protocol_manager: protocolManagerConfigSchema.optional(),
	semantic_cache: cacheConfigSchema.optional(),
	prompt_cache: promptCacheConfigSchema.optional(),
	fallback: fallbackConfigSchema.optional(),
});

// Usage schema
export const adaptiveUsageSchema = z.object({
	prompt_tokens: z.number(),
	completion_tokens: z.number(),
	total_tokens: z.number(),
	cache_tier: cacheTierSchema.optional(),
});

// Chat completion choice schema
export const chatCompletionChoiceSchema = z.object({
	index: z.number(),
	message: chatCompletionMessageSchema,
	finish_reason: z
		.enum(["stop", "length", "tool_calls", "content_filter", "function_call"])
		.nullable(),
	logprobs: z.any().optional(),
});

// Chat completion response schema
export const adaptiveChatCompletionResponseSchema = z.object({
	id: z.string(),
	choices: z.array(chatCompletionChoiceSchema),
	created: z.number(),
	model: z.string(),
	object: z.string(),
	service_tier: z.enum(["auto", "default", "flex"]).optional(),
	system_fingerprint: z.string().optional(),
	usage: adaptiveUsageSchema,
	provider: z.string().optional(),
});

// Chat completion chunk choice schema
export const chatCompletionChunkChoiceSchema = z.object({
	index: z.number(),
	delta: z.object({
		role: z
			.enum(["system", "user", "assistant", "tool", "function"])
			.optional(),
		content: z.string().optional(),
		tool_calls: z.array(z.any()).optional(),
		function_call: z.any().optional(),
	}),
	finish_reason: z
		.enum(["stop", "length", "tool_calls", "content_filter", "function_call"])
		.nullable(),
	logprobs: z.any().optional(),
});

// Chat completion chunk schema
export const adaptiveChatCompletionChunkSchema = z.object({
	id: z.string(),
	choices: z.array(chatCompletionChunkChoiceSchema),
	created: z.number(),
	model: z.string(),
	object: z.string(),
	service_tier: z.enum(["auto", "default", "flex"]).optional(),
	system_fingerprint: z.string().optional(),
	usage: adaptiveUsageSchema.optional(),
	provider: z.string().optional(),
});

// TypeScript types derived from schemas
export type ModelCapability = z.infer<typeof modelCapabilitySchema>;
export type FallbackMode = z.infer<typeof fallbackModeSchema>;
export type FallbackConfig = z.infer<typeof fallbackConfigSchema>;
export type ProtocolManagerConfig = z.infer<typeof protocolManagerConfigSchema>;
export type CacheConfig = z.infer<typeof cacheConfigSchema>;
export type PromptCacheConfig = z.infer<typeof promptCacheConfigSchema>;
export type ChatCompletionMessage = z.infer<typeof chatCompletionMessageSchema>;
export type AdaptiveChatCompletionRequest = z.infer<
	typeof adaptiveChatCompletionRequestSchema
>;
export type AdaptiveUsage = z.infer<typeof adaptiveUsageSchema>;
export type ChatCompletionChoice = z.infer<typeof chatCompletionChoiceSchema>;
export type AdaptiveChatCompletionResponse = z.infer<
	typeof adaptiveChatCompletionResponseSchema
>;
export type ChatCompletionChunkChoice = z.infer<
	typeof chatCompletionChunkChoiceSchema
>;
export type AdaptiveChatCompletionChunk = z.infer<
	typeof adaptiveChatCompletionChunkSchema
>;
