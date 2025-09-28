import { z } from "zod";
import type {
	GenerateContentRequest,
	GenerateContentResponse,
	GenerateContentStreamResponse,
} from "@google/genai";
import type { ProviderConfig } from "./providers";

// Extended request type with Adaptive-specific fields
export interface AdaptiveGeminiRequest extends GenerateContentRequest {
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

// Zod schema for validation (including Adaptive extensions)
export const geminiGenerateContentRequestSchema = z.object({
	// Core Gemini API fields
	model: z.string(),
	contents: z.any(),
	config: z.any().optional(),

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

// Extended response types with Adaptive features - keep Gemini format
export interface AdaptiveGeminiResponse extends GenerateContentResponse {
	// Adaptive-specific extensions
	provider?: string;
	model?: string;
	cache_tier?: string;
}

// Streaming chunk type extending the official SDK streaming response
export interface AdaptiveGeminiChunk extends GenerateContentStreamResponse {
	// Adaptive-specific extensions
	provider?: string;
	model?: string;
	cache_tier?: string;
}

export type { ProviderConfig };