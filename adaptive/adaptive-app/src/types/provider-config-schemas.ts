import { z } from "zod";

// Create provider config schema
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

	// Authentication for API call
	apiKey: z.string().optional(),
});

// Update provider config schema
export const updateProviderConfigSchema = z.object({
	id: z.string(),
	displayName: z.string().min(1).max(100).optional(),
	providerApiKey: z.string().min(1, "API key is required").optional(),
	customHeaders: z.record(z.string(), z.string()).optional(),
	customSettings: z.record(z.string(), z.any()).optional(),

	// Authentication for API call
	apiKey: z.string().optional(),
});

// Get provider configs schema
export const getProviderConfigsSchema = z.object({
	projectId: z.string(),
	providerId: z.string().optional(), // Filter by specific provider
	apiKey: z.string().optional(),
});

// Get provider config by ID schema
export const providerConfigByIdSchema = z.object({
	id: z.string(),
	apiKey: z.string().optional(),
});

// Delete provider config schema
export const deleteProviderConfigSchema = z.object({
	id: z.string(),
	apiKey: z.string().optional(),
});

// Type exports
export type CreateProviderConfigInput = z.infer<
	typeof createProviderConfigSchema
>;
export type UpdateProviderConfigInput = z.infer<
	typeof updateProviderConfigSchema
>;
export type GetProviderConfigsInput = z.infer<typeof getProviderConfigsSchema>;
export type ProviderConfigByIdInput = z.infer<typeof providerConfigByIdSchema>;
export type DeleteProviderConfigInput = z.infer<
	typeof deleteProviderConfigSchema
>;
