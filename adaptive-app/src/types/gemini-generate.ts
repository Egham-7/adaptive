import type {
	GenerateContentParameters,
	GenerateContentResponse,
	GenerateContentResponseUsageMetadata,
} from "@google/genai";
import { z } from "zod";
import type { CacheTier } from "./cache";
import type { ProviderConfig } from "./providers";

// Extended request type with Adaptive-specific fields
export interface AdaptiveGeminiRequest extends GenerateContentParameters {
	// Adaptive-specific extensions (passed through to backend)
	provider_configs?: Record<string, any>;
	model_router?: Record<string, any>;
	semantic_cache?: {
		enabled: boolean;
		semantic_threshold: number;
	};
	prompt_cache?: Record<string, any>;
	fallback?: Record<string, any>;
}

// Base schema for GenerateContentParameters
const generateContentParametersSchema = z.object({
	model: z.string(),
	contents: z.any(), // ContentListUnion - complex nested type, using any for now
	config: z.any().optional(), // GenerateContentConfig - complex nested type, using any for now
});

// Adaptive extensions schema using ts-to-zod generated structure
export const geminiGenerateContentRequestSchema =
	generateContentParametersSchema.extend({
		// Adaptive-specific extensions
		provider_configs: z.record(z.string(), z.any()).optional(),
		model_router: z.record(z.string(), z.any()).optional(),
		semantic_cache: z
			.object({
				enabled: z.boolean(),
				semantic_threshold: z.number().min(0).max(1),
			})
			.optional(),
		prompt_cache: z.record(z.string(), z.any()).optional(),
		fallback: z.record(z.string(), z.any()).optional(),
	});

export type GeminiGenerateContentRequest = z.infer<
	typeof geminiGenerateContentRequestSchema
>;

// Additional Zod schemas for response types using intersection
export const adaptiveGeminiResponseSchema = z.intersection(
	z.any(), // Complex Google AI response type
	z.object({
		provider: z.string(),
	}),
);

export const adaptiveGeminiUsageSchema = z.intersection(
	z.any(), // Complex usage metadata type
	z.object({
		cache_tier: z
			.enum(["semantic_exact", "semantic_similar", "prompt_response"])
			.optional(),
	}),
);

export const adaptiveGeminiChunkSchema = z.intersection(
	z.any(), // Complex Google AI response type
	z.object({
		provider: z.string(),
		usageMetadata: adaptiveGeminiUsageSchema.optional(),
	}),
);

// Extended response types with Adaptive features - keep Gemini format
export interface AdaptiveGeminiResponse extends GenerateContentResponse {
	// Adaptive-specific extensions
	provider: string;
}

export interface AdaptiveGeminiUsage
	extends GenerateContentResponseUsageMetadata {
	cache_tier?: CacheTier;
}

// Streaming chunk type extending the official SDK streaming response
export interface AdaptiveGeminiChunk extends GenerateContentResponse {
	// Adaptive-specific extensions
	provider: string;
	usageMetadata?: AdaptiveGeminiUsage;
}

export type { ProviderConfig };
