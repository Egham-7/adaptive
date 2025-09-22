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

// Cache configuration interface
export interface CacheConfig {
	enabled: boolean;
	semantic_threshold?: number;
}

// Cache configuration Zod schema
export const cacheConfigSchema = z.object({
	enabled: z.boolean().optional(),
	semantic_threshold: z.number().optional(),
});

// Prompt cache configuration
export interface PromptCacheConfig {
	enabled?: boolean;
	ttl?: number;
}

export const promptCacheConfigSchema = z.object({
	enabled: z.boolean().optional(),
	ttl: z.number().optional(),
});

// Semantic cache configuration (alias for backward compatibility)
export interface SemanticCacheConfig extends CacheConfig {}

export const semanticCacheConfigSchema = z.object({
	enabled: z.boolean(),
	semantic_threshold: z.number().optional(),
});

// Model router cache configuration (for select model requests)
export type ModelRouterCacheConfig = z.infer<typeof cacheConfigSchema>;
