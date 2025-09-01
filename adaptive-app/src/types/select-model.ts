import { z } from "zod";

// Create Zod schema for ModelCapability (partial type for flexible model selection)
const modelCapabilitySchema = z
	.object({
		description: z.string().optional(),
		provider: z.string().optional(),
		model_name: z.string().optional(),
		cost_per_1m_input_tokens: z.number().optional(),
		cost_per_1m_output_tokens: z.number().optional(),
		max_context_tokens: z.number().optional(),
		max_output_tokens: z.number().optional(),
		supports_function_calling: z.boolean().optional(),
		languages_supported: z.array(z.string()).optional(),
		model_size_params: z.string().optional(),
		latency_tier: z.string().optional(),
		task_type: z.string().optional(),
		complexity: z.string().optional(),
	})
	.refine(
		(data) => {
			// Require at least one field to be defined
			const values = Object.values(data);
			return values.some((value) => value !== undefined);
		},
		{
			message: "At least one model capability attribute must be provided",
		},
	);

// Create Zod schema for ModelRouterConfig
const modelRouterConfigSchema = z.object({
	models: z.array(modelCapabilitySchema).optional(),
	cost_bias: z.number().optional(),
	complexity_threshold: z.number().optional(),
	token_threshold: z.number().optional(),
});

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
		model_router: modelRouterConfigSchema.optional(),
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
