import { z } from "zod";

// Model capability schema matching Go backend ModelCapability struct
export const modelCapabilitySchema = z.object({
	description: z.string().optional(),
	maxContextTokens: z.number().min(1).optional(),
	maxOutputTokens: z.number().min(1).optional(),
	supportsFunctionCalling: z.boolean().default(false),
	languagesSupported: z.array(z.string()).optional(),
	modelSizeParams: z.string().optional(),
	latencyTier: z.enum(["low", "medium", "high"]).optional(),
	taskType: z.string().optional(),
	complexity: z.enum(["low", "medium", "high"]).optional(),
});

// Provider model schema for creating models within a provider
export const providerModelSchema = z.object({
	name: z.string().min(1, "Model name is required"),
	displayName: z.string().min(1, "Display name is required"),
	type: z
		.enum(["completion", "chat", "embedding", "image", "audio"])
		.default("chat"),
	inputTokenCost: z.number().min(0, "Input token cost must be non-negative"),
	outputTokenCost: z.number().min(0, "Output token cost must be non-negative"),
	capabilities: modelCapabilitySchema.optional(),
});

// Create provider schema
export const createProviderSchema = z.object({
	projectId: z.string(),
	name: z
		.string()
		.min(1, "Provider name is required")
		.max(50, "Provider name too long")
		.regex(
			/^[a-z0-9-_]+$/,
			"Provider name must be lowercase alphanumeric with hyphens and underscores only",
		),
	displayName: z.string().min(1, "Display name is required").max(100),
	description: z.string().max(500).optional(),
	visibility: z
		.enum(["project", "organization", "community"])
		.default("project"),

	// Custom provider configuration
	baseUrl: z.string().url("Must be a valid URL").optional(),
	authType: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	authHeaderName: z.string().max(100).optional(),
	apiKey: z
		.string()
		.min(1, "API key is required for custom providers")
		.optional(),
	healthEndpoint: z.string().max(200).optional(),
	rateLimitRpm: z.number().min(1).max(100000).optional(),
	timeoutMs: z.number().min(1000).max(120000).optional(),
	retryConfig: z.record(z.string(), z.any()).optional(),
	headers: z.record(z.string(), z.string()).optional(),

	models: z.array(providerModelSchema).min(1, "At least one model is required"),
});

// Update provider schema
export const updateProviderSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	description: z.string().max(500).optional(),
	visibility: z
		.enum(["project", "organization", "community"])
		.optional(),

	// Custom provider configuration updates
	baseUrl: z.string().url("Must be a valid URL").optional(),
	authType: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	authHeaderName: z.string().max(100).optional(),
	apiKey: z
		.string()
		.min(1, "API key is required for custom providers")
		.optional(),
	healthEndpoint: z.string().max(200).optional(),
	rateLimitRpm: z.number().min(1).max(100000).optional(),
	timeoutMs: z.number().min(1000).max(120000).optional(),
	retryConfig: z.record(z.string(), z.any()).optional(),
	headers: z.record(z.string(), z.string()).optional(),

	isActive: z.boolean().optional(),
});

// Add model to existing provider schema
export const addProviderModelSchema = z.object({
	providerId: z.string(),
	name: z.string().min(1, "Model name is required"),
	displayName: z.string().min(1, "Display name is required"),
	type: z
		.enum(["completion", "chat", "embedding", "image", "audio"])
		.default("chat"),
	inputTokenCost: z.number().min(0, "Input token cost must be non-negative"),
	outputTokenCost: z.number().min(0, "Output token cost must be non-negative"),
	capabilities: modelCapabilitySchema.optional(),
	apiKey: z.string().optional(),
});

// Update provider model schema
export const updateProviderModelSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	type: z
		.enum(["completion", "chat", "embedding", "image", "audio"])
		.optional(),
	inputTokenCost: z.number().min(0).optional(),
	outputTokenCost: z.number().min(0).optional(),
	capabilities: modelCapabilitySchema.optional(),
	isActive: z.boolean().optional(),
	apiKey: z.string().optional(),
});

// Input schemas for API endpoints
export const providerByIdSchema = z.object({
	id: z.string(),
	apiKey: z.string().optional(),
});

export const providerByNameSchema = z.object({
	name: z
		.string()
		.min(1, "Provider name is required")
		.max(50, "Provider name too long")
		.regex(
			/^[a-z0-9-_]+$/,
			"Provider name must be lowercase alphanumeric with hyphens and underscores only",
		),
	apiKey: z.string().optional(),
});

export const getAllProvidersSchema = z.object({
	projectId: z.string().optional(), // Required to see project/org providers
	apiKey: z.string().optional(),
});

// Type exports
export type CreateProviderInput = z.infer<typeof createProviderSchema>;
export type UpdateProviderInput = z.infer<typeof updateProviderSchema>;
export type AddProviderModelInput = z.infer<typeof addProviderModelSchema>;
export type UpdateProviderModelInput = z.infer<
	typeof updateProviderModelSchema
>;
export type ProviderModel = z.infer<typeof providerModelSchema>;
