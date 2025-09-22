import { z } from "zod";
import type { Prisma } from "../../prisma/generated";

// Define supported providers as a union type
export type Provider =
	| "openai"
	| "anthropic"
	| "gemini"
	| "groq"
	| "deepseek"
	| "huggingface"
	| "grok"
	| "adaptive"
	| null;

// Define supported authentication types
export type AuthType = "bearer" | "api_key" | "basic" | "custom";

// Provider config interface
export interface ProviderConfig {
	api_key?: string;
	base_url?: string;
	auth_type?: AuthType;
	auth_header_name?: string;
	health_endpoint?: string;
	rate_limit_rpm?: number;
	timeout_ms?: number;
	retry_config?: Record<string, unknown>;
	headers?: Record<string, string>;
}

// Provider config Zod schema
export const providerConfigSchema = z.object({
	api_key: z.string().optional(),
	base_url: z.string().optional(),
	auth_type: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	auth_header_name: z.string().optional(),
	health_endpoint: z.string().optional(),
	rate_limit_rpm: z.number().optional(),
	timeout_ms: z.number().optional(),
	retry_config: z.record(z.string(), z.unknown()).optional(),
	headers: z.record(z.string(), z.string()).optional(),
});

// Fallback configuration
export type FallbackMode = "sequential" | "race";

export interface FallbackConfig {
	enabled?: boolean;
	mode?: FallbackMode;
}

export const fallbackConfigSchema = z.object({
	enabled: z.boolean().optional(),
	mode: z.enum(["sequential", "race"]).optional(),
});

// Model capability schema matching Go backend ModelCapability struct
export const modelCapabilitySchemaBackend = z.object({
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
	inputTokenCost: z.number().min(0, "Input token cost must be non-negative"),
	outputTokenCost: z.number().min(0, "Output token cost must be non-negative"),
	capabilities: modelCapabilitySchemaBackend.optional(),
});

// Create provider schema (template without API keys)
export const createProviderSchema = z.object({
	projectId: z.string().optional(), // Optional for system/community providers
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
		.enum(["system", "project", "organization", "community"])
		.default("project"),

	// Provider template configuration (no secrets)
	baseUrl: z.string().optional(),
	authType: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	authHeaderName: z.string().max(100).optional(),
	healthEndpoint: z.string().max(200).optional(),
	rateLimitRpm: z.number().min(1).max(100000).optional(),
	timeoutMs: z.number().min(1000).max(120000).optional(),
	retryConfig: z.record(z.string(), z.any()).optional(),
	headers: z.record(z.string(), z.string()).optional(),

	models: z.array(providerModelSchema).min(1, "At least one model is required"),

	// Authentication for API call
	apiKey: z.string().optional(),
});

// Update provider schema (template updates only)
export const updateProviderSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	description: z.string().max(500).optional(),
	visibility: z
		.enum(["system", "project", "organization", "community"])
		.optional(),

	// Provider template configuration updates (no secrets)
	baseUrl: z.string().optional(),
	authType: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	authHeaderName: z.string().max(100).optional(),
	healthEndpoint: z.string().max(200).optional(),
	rateLimitRpm: z.number().min(1).max(100000).optional(),
	timeoutMs: z.number().min(1000).max(120000).optional(),
	retryConfig: z.record(z.string(), z.any()).optional(),
	headers: z.record(z.string(), z.string()).optional(),

	// Authentication for API call
	apiKey: z.string().optional(),
});

// Add model to existing provider schema
export const addProviderModelSchema = z.object({
	providerId: z.string(),
	name: z.string().min(1, "Model name is required"),
	displayName: z.string().min(1, "Display name is required"),
	inputTokenCost: z.number().min(0, "Input token cost must be non-negative"),
	outputTokenCost: z.number().min(0, "Output token cost must be non-negative"),
	capabilities: modelCapabilitySchemaBackend.optional(),
	apiKey: z.string().optional(),
});

// Update provider model schema
export const updateProviderModelSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	inputTokenCost: z.number().min(0).optional(),
	outputTokenCost: z.number().min(0).optional(),
	capabilities: modelCapabilitySchemaBackend.optional(),
	apiKey: z.string().optional(),
});

// Provider config management schemas (for database operations)
export const createProviderConfigSchema = z.object({
	projectId: z.string(),
	providerId: z.string(),
	displayName: z
		.string()
		.min(1, "Display name is required")
		.max(100)
		.optional(),
	providerApiKey: z.string().min(1, "API key is required"),
	customHeaders: z.record(z.string(), z.string()).optional(),
	customSettings: z.record(z.string(), z.any()).optional(),
	apiKey: z.string().optional(),
});

export const updateProviderConfigSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	providerApiKey: z.string().min(1, "API key is required").optional(),
	customHeaders: z.record(z.string(), z.string()).optional(),
	customSettings: z.record(z.string(), z.any()).optional(),
	apiKey: z.string().optional(),
});

export const getProviderConfigsSchema = z.object({
	projectId: z.string(),
	providerId: z.string().optional(),
	apiKey: z.string().optional(),
});

export const providerConfigByIdSchema = z.object({
	id: z.string(),
	apiKey: z.string().optional(),
});

export const deleteProviderConfigSchema = z.object({
	id: z.string(),
	apiKey: z.string().optional(),
});

// Input schemas for API endpoints
export const providerByIdSchema = z.object({
	id: z.string(),
	projectId: z.string().optional(), // Required to see project/org scoped providers
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
	projectId: z.string().optional(), // Required to see project/org scoped providers
	apiKey: z.string().optional(),
});

export const getAllProvidersSchema = z.object({
	projectId: z.string().optional(), // Required to see project/org providers
	apiKey: z.string().optional(),
});

// Prisma payload types for providers
export type ProviderWithModels = Prisma.ProviderGetPayload<{
	include: {
		models: {
			include: {
				capabilities: true;
			};
		};
	};
}>;

// Type exports from schemas
export type CreateProviderInput = z.infer<typeof createProviderSchema>;
export type UpdateProviderInput = z.infer<typeof updateProviderSchema>;
export type AddProviderModelInput = z.infer<typeof addProviderModelSchema>;
export type UpdateProviderModelInput = z.infer<
	typeof updateProviderModelSchema
>;
export type ProviderModel = z.infer<typeof providerModelSchema>;

export type CreateProviderConfig = z.infer<typeof createProviderConfigSchema>;
export type UpdateProviderConfig = z.infer<typeof updateProviderConfigSchema>;
export type GetProviderConfigs = z.infer<typeof getProviderConfigsSchema>;
export type ProviderConfigByIdInput = z.infer<typeof providerConfigByIdSchema>;
export type DeleteProviderConfig = z.infer<typeof deleteProviderConfigSchema>;
