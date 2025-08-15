import type {
	ChatCompletion as OpenAIChatCompletion,
	ChatCompletionChunk as OpenAIChatCompletionChunk,
} from "openai/resources/chat/completions";
import { z } from "zod";

// Define supported providers as a union type
export type Provider =
	| "openai"
	| "anthropic"
	| "gemini"
	| "groq"
	| "deepseek"
	| "huggingface"
	| "grok"
	| "adaptive"
	| null;

export const providerConfigSchema = z.object({
	api_key: z.string().optional(),
	base_url: z.string().url("Must be a valid URL").optional(),
	auth_type: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	auth_header_name: z.string().max(100).optional(),
	health_endpoint: z.string().max(200).optional(),
	rate_limit_rpm: z.number().int().min(1).max(100000).optional(),
	timeout_ms: z.number().int().min(1000).max(120000).optional(),
	retry_config: z.record(z.string(), z.any()).optional(),
	headers: z.record(z.string(), z.string()).optional(),
});

// Infer TypeScript type from Zod schema
export type ProviderConfig = z.infer<typeof providerConfigSchema>;

// Chat completion request schema with our custom extensions
export const chatCompletionRequestSchema = z.object({
	// OpenAI-compatible fields (we don't validate these deeply since they come from external lib)
	messages: z.array(z.any()),
	model: z.string(),
	frequency_penalty: z.number().min(-2).max(2).optional(),
	logprobs: z.boolean().optional(),
	max_completion_tokens: z.number().min(1).optional(),
	max_tokens: z.number().min(1).optional(),
	n: z.number().min(1).max(128).optional(),
	presence_penalty: z.number().min(-2).max(2).optional(),
	seed: z.number().optional(),
	store: z.boolean().optional(),
	temperature: z.number().min(0).max(2).optional(),
	top_logprobs: z.number().min(0).max(20).optional(),
	top_p: z.number().min(0).max(1).optional(),
	parallel_tool_calls: z.boolean().optional(),
	user: z.string().optional(),
	audio: z.any().optional(),
	logit_bias: z.record(z.string(), z.number()).optional(),
	metadata: z.record(z.string(), z.string()).optional(),
	modalities: z.array(z.enum(["text", "audio"])).optional(),
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
	stream: z.boolean().optional(),

	// Our custom extensions
	provider_configs: z.record(z.string(), providerConfigSchema).optional(),

	// Adaptive system extensions
	protocol_manager: z
		.object({
			models: z.array(z.any()).optional(),
			cost_bias: z.number().min(0).max(1).optional(),
			complexity_threshold: z.number().optional(),
			token_threshold: z.number().int().optional(),
		})
		.optional(),
	semantic_cache: z
		.object({
			enabled: z.boolean(),
			semantic_threshold: z.number().optional(),
		})
		.optional(),
	prompt_cache: z
		.object({
			enabled: z.boolean().optional(),
			ttl: z.number().int().optional(),
		})
		.optional(),
	fallback: z
		.object({
			enabled: z.boolean().optional(),
			mode: z.enum(["sequential", "parallel"]).optional(),
		})
		.optional(),
});

// Infer TypeScript type from Zod schema
export type ChatCompletionRequest = z.infer<typeof chatCompletionRequestSchema>;

// Extend OpenAI types with provider field
export interface ChatCompletion extends OpenAIChatCompletion {
	provider: Provider;
}

export interface ChatCompletionChunk extends OpenAIChatCompletionChunk {
	provider: string;
}
