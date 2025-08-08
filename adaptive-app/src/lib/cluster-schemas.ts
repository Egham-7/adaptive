import { z } from "zod";

// Shared Zod schemas for LLM clusters - single source of truth

export const clusterModelSchema = z.object({
	provider: z.string().min(1, "Provider is required"),
	modelName: z.string().min(1),
	priority: z.number().min(1).default(1),
});

export const createClusterSchema = z.object({
	projectId: z.string(),
	name: z.string().min(1, "Cluster name is required").max(50),
	description: z.string().max(500).optional(),
	fallbackEnabled: z.boolean().default(true),
	fallbackMode: z
		.enum(["sequential", "parallel", "disabled"])
		.default("parallel"),
	enableCircuitBreaker: z.boolean().default(true),
	maxRetries: z.number().min(1).max(10).default(3),
	timeoutMs: z.number().min(1000).max(120000).default(30000),
	costBias: z.number().min(0).max(1).default(0.5),
	complexityThreshold: z.number().min(0).max(1).optional(),
	tokenThreshold: z.number().min(1).optional(),
	enableSemanticCache: z.boolean().default(true),
	semanticThreshold: z.number().min(0).max(1).default(0.85),
	enablePromptCache: z.boolean().default(true),
	promptCacheTTL: z.number().min(60).max(86400).default(3600),
	apiKey: z.string().optional(),
	models: z.array(clusterModelSchema).min(1, "At least one model is required"),
});

export const updateClusterSchema = z.object({
	id: z.string(),
	name: z.string().min(1).max(50).optional(),
	description: z.string().max(500).optional(),
	fallbackEnabled: z.boolean().optional(),
	fallbackMode: z.enum(["sequential", "parallel", "disabled"]).optional(),
	enableCircuitBreaker: z.boolean().optional(),
	maxRetries: z.number().min(1).max(10).optional(),
	timeoutMs: z.number().min(1000).max(120000).optional(),
	costBias: z.number().min(0).max(1).optional(),
	complexityThreshold: z.number().min(0).max(1).optional(),
	tokenThreshold: z.number().min(1).optional(),
	enableSemanticCache: z.boolean().optional(),
	semanticThreshold: z.number().min(0).max(1).optional(),
	enablePromptCache: z.boolean().optional(),
	promptCacheTTL: z.number().min(60).max(86400).optional(),
	isActive: z.boolean().optional(),
	apiKey: z.string().optional(),
});

export const addModelSchema = z.object({
	clusterId: z.string(),
	provider: clusterModelSchema.shape.provider,
	modelName: z.string(),
	priority: z.number().min(1).default(1),
	apiKey: z.string().optional(),
});

// Input schemas for API endpoints
export const projectClusterParamsSchema = z.object({
	projectId: z.string(),
	apiKey: z.string().optional(),
});

export const clusterByNameParamsSchema = z.object({
	projectId: z.string(),
	name: z.string(),
	apiKey: z.string().optional(),
});

// Type exports
export type CreateClusterInput = z.infer<typeof createClusterSchema>;
export type UpdateClusterInput = z.infer<typeof updateClusterSchema>;
export type AddModelInput = z.infer<typeof addModelSchema>;
export type ClusterModel = z.infer<typeof clusterModelSchema>;
