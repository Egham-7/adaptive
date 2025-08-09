import { z } from "zod";

// Simple cluster provider schema
export const clusterProviderSchema = z.object({
	providerId: z.string().min(1, "Provider ID is required"),
	configId: z.string().optional(), // Updated from providerConfigId
});

// Full cluster creation schema with all cluster configuration fields
export const createClusterSchema = z.object({
	projectId: z.string().min(1, "Project ID is required"),
	name: z.string().min(1, "Cluster name is required").max(50),
	description: z.string().max(500).optional(),

	// Fallback Config fields
	fallbackEnabled: z.boolean().default(true),
	fallbackMode: z.enum(["sequential", "parallel"]).default("parallel"),

	// Protocol Manager Config fields
	enableCircuitBreaker: z.boolean().default(true),
	maxRetries: z.number().int().min(0).max(10).default(3),
	timeoutMs: z.number().int().min(1000).max(300000).default(30000),
	costBias: z.number().min(0).max(1).default(0.5),
	complexityThreshold: z.number().min(0).max(1).optional(),
	tokenThreshold: z.number().int().min(1).optional(),

	// Semantic Cache Config
	enableSemanticCache: z.boolean().default(true),
	semanticThreshold: z.number().min(0).max(1).default(0.85),

	// Prompt Cache Config
	enablePromptCache: z.boolean().default(true),
	promptCacheTTL: z.number().int().min(1).default(3600),

	providers: z
		.array(clusterProviderSchema)
		.min(1, "At least one provider is required"),
	apiKey: z.string().optional(),
});

// Simple cluster update schema
export const updateClusterSchema = z.object({
	id: z.string().min(1, "Cluster ID is required"),
	name: z.string().min(1).max(50).optional(),
	description: z.string().max(500).optional(),

	// Fallback Config fields
	fallbackEnabled: z.boolean().optional(),
	fallbackMode: z.enum(["sequential", "parallel"]).optional(),

	// Protocol Manager Config fields
	enableCircuitBreaker: z.boolean().optional(),
	maxRetries: z.number().int().min(0).max(10).optional(),
	timeoutMs: z.number().int().min(1000).max(300000).optional(),
	costBias: z.number().min(0).max(1).optional(),
	complexityThreshold: z.number().min(0).max(1).optional(),
	tokenThreshold: z.number().int().min(1).optional(),

	// Semantic Cache Config
	enableSemanticCache: z.boolean().optional(),
	semanticThreshold: z.number().min(0).max(1).optional(),

	// Prompt Cache Config
	enablePromptCache: z.boolean().optional(),
	promptCacheTTL: z.number().int().min(1).optional(),

	apiKey: z.string().optional(),
});

// Route parameter schemas
export const projectClusterParamsSchema = z.object({
	projectId: z.string().min(1, "Project ID is required"),
	apiKey: z.string().optional(),
});

export const clusterByNameParamsSchema = z.object({
	projectId: z.string().min(1, "Project ID is required"),
	name: z.string().min(1, "Cluster name is required"),
	apiKey: z.string().optional(),
});

// Add provider to cluster schema
export const addProviderToClusterSchema = z.object({
	clusterId: z.string().min(1, "Cluster ID is required"),
	providerId: z.string().min(1, "Provider ID is required"),
	configId: z.string().optional(), // Updated from providerConfigId
	apiKey: z.string().optional(),
});

// Type exports
export type CreateClusterInput = z.infer<typeof createClusterSchema>;
export type UpdateClusterInput = z.infer<typeof updateClusterSchema>;
export type AddProviderToClusterInput = z.infer<
	typeof addProviderToClusterSchema
>;
export type ClusterProvider = z.infer<typeof clusterProviderSchema>;
