import { z } from "zod";
import { cacheTierSchema } from "@/types/cache";
import { providerEnum } from "@/types/providers";

/**
 * Usage input schema for API usage recording
 */
export const usageInputSchema = z.object({
	promptTokens: z.number().min(0, "Prompt tokens must be non-negative"),
	completionTokens: z.number().min(0, "Completion tokens must be non-negative"),
	totalTokens: z.number().min(0, "Total tokens must be non-negative"),
});

/**
 * Schema for recording API usage
 */
export const recordApiUsageInputSchema = z.object({
	apiKey: z.string().min(1, "API key is required"),
	provider: z.enum(providerEnum).nullable(),
	model: z.string().nullable(),
	usage: usageInputSchema,
	duration: z.number().min(0, "Duration must be non-negative"),
	timestamp: z.date(),
	requestCount: z.number().min(1).default(1),
	clusterId: z.string().optional(),
	metadata: z.record(z.string(), z.any()).optional(),
	error: z.string().optional(),
	cacheTier: cacheTierSchema.optional(),
});

/**
 * Schema for recording API errors
 */
export const recordErrorInputSchema = z.object({
	apiKey: z.string().min(1, "API key is required"),
	provider: z.enum(providerEnum).optional(),
	model: z.string().optional(),
	error: z.string().min(1, "Error message is required"),
	timestamp: z.date(),
});

// Type inference from schemas
export type UsageInput = z.infer<typeof usageInputSchema>;
export type RecordApiUsageInput = z.infer<typeof recordApiUsageInputSchema>;
export type RecordErrorInput = z.infer<typeof recordErrorInputSchema>;

/**
 * Function to create usage metadata
 */
export function createUsageMetadata(
	input: RecordApiUsageInput,
	apiKey: { userId: string; [key: string]: any },
) {
	return {
		...input.metadata,
		duration: input.duration,
		timestamp: input.timestamp,
		error: input.error,
		userId: apiKey.userId,
		cacheTier: input.cacheTier,
	};
}
