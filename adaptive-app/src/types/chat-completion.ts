import type {
	ChatCompletionCreateParamsBase,
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

// Provider config type
export interface ProviderConfig {
	api_key?: string;
	base_url?: string;
	auth_type?: "bearer" | "api_key" | "basic" | "custom";
	auth_header_name?: string;
	health_endpoint?: string;
	rate_limit_rpm?: number;
	timeout_ms?: number;
	retry_config?: Record<string, unknown>;
	headers?: Record<string, string>;
}

// Model capability type
export interface ModelCapability {
	description?: string;
	provider: string;
	model_name: string;
	cost_per_1m_input_tokens: number;
	cost_per_1m_output_tokens: number;
	max_context_tokens: number;
	max_output_tokens?: number;
	supports_function_calling: boolean;
	languages_supported?: string[];
	model_size_params?: string;
	latency_tier?: string;
	task_type?: string;
	complexity?: string;
}

// Fallback types
export type FallbackMode = "sequential" | "race";

export interface FallbackConfig {
	enabled?: boolean;
	mode?: FallbackMode;
}

// Model router config type
export interface ModelRouterConfig {
	models?: ModelCapability[];
	cost_bias?: number;
	complexity_threshold?: number;
	token_threshold?: number;
}

// Cache config types
export interface CacheConfig {
	enabled: boolean;
	semantic_threshold?: number;
}

export interface PromptCacheConfig {
	enabled?: boolean;
	ttl?: number;
}

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
	provider?: Provider;
	usage?: AdaptiveUsage;
	cache_tier?: CacheTier;
}

export interface ChatCompletionChunk
	extends Omit<OpenAIChatCompletionChunk, "usage"> {
	provider?: Provider;
	usage?: AdaptiveUsage;
}
