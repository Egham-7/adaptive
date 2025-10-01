import type {
	ChatCompletionCreateParamsBase,
	ChatCompletion as OpenAIChatCompletion,
	ChatCompletionChunk as OpenAIChatCompletionChunk,
} from "openai/resources/chat/completions";
import { z } from "zod";
import type { CacheConfig, CacheTier, PromptCacheConfig } from "./cache";
import type { ModelRouterConfig } from "./models";
import type { FallbackConfig, ProviderConfig, ProviderType } from "./providers";

// Extend OpenAI's ChatCompletionCreateParamsBase with our custom fields
export interface ChatCompletionRequest extends ChatCompletionCreateParamsBase {
	// Our custom extensions
	provider_configs?: Record<string, ProviderConfig>;

	// Adaptive system extensions
	model_router?: ModelRouterConfig;
	semantic_cache?: CacheConfig;
	prompt_cache?: PromptCacheConfig;
	fallback?: FallbackConfig;
}

// Usage type
export interface AdaptiveUsage {
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
	cache_tier?: CacheTier;
}

// Extend OpenAI types with provider field and adaptive usage
export interface ChatCompletion extends Omit<OpenAIChatCompletion, "usage"> {
	provider?: ProviderType | null;
	usage?: AdaptiveUsage;
	cache_tier?: CacheTier;
}

export interface ChatCompletionChunk
	extends Omit<OpenAIChatCompletionChunk, "usage"> {
	provider?: ProviderType | null;
	usage?: AdaptiveUsage;
}

// Zod schema for ChatCompletionRequest validation
export const chatCompletionRequestSchema = z.looseObject({
	model: z.string().optional(),
	messages: z
		.array(
			z.object({
				role: z.enum(["system", "user", "assistant", "function", "tool"]),
				content: z
					.union([z.string(), z.array(z.unknown())])
					.nullable()
					.optional(),
			}),
		)
		.min(1, "At least one message is required"),
	temperature: z.number().min(0).max(2).optional(),
	top_p: z.number().min(0).max(1).optional(),
	n: z.number().positive().optional(),
	stream: z.boolean().optional(),
	stream_options: z
		.object({
			include_usage: z.boolean().optional(),
		})
		.optional(),
	stop: z.union([z.string(), z.array(z.string())]).optional(),
	max_tokens: z.number().positive().optional(),
	presence_penalty: z.number().min(-2).max(2).optional(),
	frequency_penalty: z.number().min(-2).max(2).optional(),
	logit_bias: z.record(z.string(), z.number()).optional(),
	user: z.string().optional(),
}); // Allow additional fields for custom extensions
