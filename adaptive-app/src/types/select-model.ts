import { z } from "zod";
import { adaptiveChatCompletionRequestSchema } from "./chat-completion";

// Select model request uses the same schema as adaptive chat completion request
export const selectModelRequestSchema = adaptiveChatCompletionRequestSchema;

// Zod schema for selection metadata
export const selectionMetadataSchema = z.object({
	provider: z.string(),
	model: z.string(),
	reasoning: z.string().optional(),
	cost_per_1m_tokens: z.number().optional(),
	complexity: z.string().optional(),
	cache_source: z.string().optional(),
});

// Zod schema for select model response
export const selectModelResponseSchema = z.object({
	request: selectModelRequestSchema,
	metadata: selectionMetadataSchema,
});

// TypeScript types derived from Zod schemas
export type SelectModelRequest = z.infer<typeof selectModelRequestSchema>;
export type SelectionMetadata = z.infer<typeof selectionMetadataSchema>;
export type SelectModelResponse = z.infer<typeof selectModelResponseSchema>;
