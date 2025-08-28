import { z } from "zod";
import {
	modelCapabilitySchema,
	modelRouterConfigSchema,
} from "./chat-completion";

// Provider-agnostic select model request schema
export const selectModelRequestSchema = z
	.object({
		// Available models with their capabilities and constraints
		models: z
			.array(modelCapabilitySchema)
			.min(1, "At least one model is required"),
		// The prompt text to analyze for optimal model selection
		prompt: z.string().min(1, "Prompt cannot be empty"),
		// Optional user identifier for tracking and personalization
		user: z.string().optional(),
		// Model router configuration for routing decisions
		model_router_config: modelRouterConfigSchema.optional(),
	})
	.strict();

// Zod schema for alternative provider/model combinations
export const alternativeSchema = z.object({
	provider: z.string(),
	model: z.string(),
});

// Zod schema for selection metadata
export const selectionMetadataSchema = z.object({
	reasoning: z.string().optional(),
	cost_per_1m_tokens: z.number().optional(),
	complexity: z.string().optional(),
	cache_source: z.string().optional(),
});

// Zod schema for select model response
export const selectModelResponseSchema = z.object({
	// Selected provider
	provider: z.string(),
	// Selected model
	model: z.string(),
	// Alternative provider/model combinations
	alternatives: z.array(alternativeSchema).optional(),
	// Additional metadata about the selection
	metadata: selectionMetadataSchema,
});

// TypeScript types derived from Zod schemas
export type SelectModelRequest = z.infer<typeof selectModelRequestSchema>;
export type Alternative = z.infer<typeof alternativeSchema>;
export type SelectionMetadata = z.infer<typeof selectionMetadataSchema>;
export type SelectModelResponse = z.infer<typeof selectModelResponseSchema>;
