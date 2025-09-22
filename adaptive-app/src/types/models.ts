import { z } from "zod";
import type { RouterOutputs } from "@/trpc/react";

// Model types inferred from tRPC router outputs
export type ModelWithMetadata =
	RouterOutputs["modelPricing"]["getAllModelsWithMetadata"][number];
export type ModelsByProvider =
	RouterOutputs["modelPricing"]["getModelsByProvider"];
export type CostComparison =
	RouterOutputs["modelPricing"]["calculateCostComparison"];

// Model capability interface
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

// Model capability Zod schema (partial for flexible model selection)
export const modelCapabilitySchema = z
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

// Model router configuration
export interface ModelRouterConfig {
	models?: ModelCapability[];
	cost_bias?: number;
	complexity_threshold?: number;
	token_threshold?: number;
}

export const modelRouterConfigSchema = z.object({
	models: z.array(modelCapabilitySchema).optional(),
	cost_bias: z.number().min(0).max(1).optional(),
	complexity_threshold: z.number().optional(),
	token_threshold: z.number().optional(),
});

// Alternative model schema
export const alternativeSchema = z.object({
	provider: z.string(),
	model: z.string(),
});

export type Alternative = z.infer<typeof alternativeSchema>;
