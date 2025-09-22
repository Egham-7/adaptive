import { z } from "zod";
import { cacheConfigSchema } from "./cache";
import { alternativeSchema, modelCapabilitySchema } from "./models";
import { toolCallSchema, toolSchema } from "./tools";

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
export type SelectModelResponse = z.infer<typeof selectModelResponseSchema>;
