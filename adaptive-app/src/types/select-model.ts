import { z } from "zod";

// Create Zod schema for CacheConfig (only configurable fields)
const cacheConfigSchema = z.object({
	enabled: z.boolean().optional(),
	semantic_threshold: z.number().optional(),
});

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

// Tool definition schema
const toolSchema = z.object({
	type: z.literal("function"),
	function: z.object({
		name: z.string(),
		description: z.string().optional(),
		parameters: z.record(z.string(), z.any()).optional(),
	}),
});

// Tool call schema
const toolCallSchema = z.object({
	id: z.string(),
	type: z.literal("function"),
	function: z.object({
		name: z.string(),
		arguments: z.string(),
	}),
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
		// Cost bias for balancing cost vs performance (0.0 = cheapest, 1.0 = best)
		cost_bias: z.number().min(0).max(1).optional(),
		// Model router cache configuration
		model_router_cache: cacheConfigSchema.optional(),
		// Tool definitions for function calling detection
		tools: z.array(toolSchema).optional(),
		// Current tool call being made (if any)
		tool_call: toolCallSchema.optional(),
	})
	.strict();

// Zod schema for alternative provider/model combinations
export const alternativeSchema = z.object({
	provider: z.string(),
	model: z.string(),
});

// Zod schema for select model response
export const selectModelResponseSchema = z.object({
	// Selected provider
	provider: z.string(),
	// Selected model
	model: z.string(),
	// Alternative provider/model combinations
	alternatives: z.array(alternativeSchema).optional(),
});

// TypeScript types derived from Zod schemas
export type SelectModelRequest = z.infer<typeof selectModelRequestSchema>;
export type Alternative = z.infer<typeof alternativeSchema>;
export type SelectModelResponse = z.infer<typeof selectModelResponseSchema>;
export type Tool = z.infer<typeof toolSchema>;
export type ToolCall = z.infer<typeof toolCallSchema>;
export type CacheConfig = z.infer<typeof cacheConfigSchema>;
