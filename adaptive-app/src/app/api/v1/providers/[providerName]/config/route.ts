import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import type { NextRequest } from "next/server";
import { z } from "zod";
import {
	createAuthError,
	createErrorResponse,
	createSuccessResponse,
	extractApiKey,
} from "@/lib/api-auth";
import { invalidateProviderConfigCache, withCache } from "@/lib/cache-utils";
import { api } from "@/trpc/server";

// Schema for provider config operations
const createConfigSchema = z.object({
	projectId: z.string(),
	providerApiKey: z.string().min(1, "API key is required"),
	displayName: z.string().min(1).max(100).optional(),
	customHeaders: z.record(z.string(), z.string()).optional(),
	customSettings: z.record(z.string(), z.any()).optional(),
});

const updateConfigSchema = z.object({
	providerApiKey: z.string().min(1, "API key is required").optional(),
	displayName: z.string().min(1).max(100).optional(),
	customHeaders: z.record(z.string(), z.string()).optional(),
	customSettings: z.record(z.string(), z.any()).optional(),
	isActive: z.boolean().optional(),
});

// POST /api/v1/providers/{providerName}/config - Create config for provider
export async function POST(
	req: NextRequest,
	{ params }: { params: Promise<{ providerName: string }> },
) {
	try {
		const { providerName } = await params;
		const body = await req.json();

		// Extract API key from headers
		const apiKey = extractApiKey(req);
		if (!apiKey) {
			return createAuthError();
		}

		// Validate request body
		const validationResult = createConfigSchema.safeParse(body);
		if (!validationResult.success) {
			return createErrorResponse(
				"Invalid request format",
				400,
				validationResult.error.issues,
			);
		}

		const { projectId, ...configData } = validationResult.data;

		// Find provider by name with caching
		const provider = await withCache(
			`provider:${providerName}:${projectId}`,
			() => api.providers.getByName({ name: providerName, projectId, apiKey }),
			300, // 5 minutes cache
		);

		if (!provider) {
			return createErrorResponse(`Provider '${providerName}' not found`, 404);
		}

		// Create provider config
		const config = await api.providerConfigs.create({
			projectId,
			providerId: provider.id,
			...configData,
			apiKey,
		});

		// Invalidate provider config cache
		await invalidateProviderConfigCache(projectId, provider.id);

		return createSuccessResponse(config, 201);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Create provider config API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}

// GET /api/v1/providers/{providerName}/config - Get config for provider
export async function GET(
	req: NextRequest,
	{ params }: { params: Promise<{ providerName: string }> },
) {
	try {
		const { providerName } = await params;
		const url = new URL(req.url);
		const projectId = url.searchParams.get("project_id");

		// Extract API key from headers
		const apiKey = extractApiKey(req);
		if (!apiKey) {
			return createAuthError();
		}

		if (!projectId) {
			return createErrorResponse("project_id parameter is required", 400);
		}

		// Find provider by name with caching
		const provider = await withCache(
			`provider:${providerName}:${projectId}`,
			() => api.providers.getByName({ name: providerName, projectId, apiKey }),
			300, // 5 minutes cache
		);

		if (!provider) {
			return createErrorResponse(`Provider '${providerName}' not found`, 404);
		}

		// Get provider configs for this project and provider with caching
		const configs = await withCache(
			`provider-config:${projectId}:${provider.id}`,
			() =>
				api.providerConfigs.getAll({
					projectId,
					providerId: provider.id,
					apiKey,
				}),
			60, // 1 minute cache for user configs
		);

		if (configs.length === 0) {
			return createErrorResponse(
				`No configuration found for provider '${providerName}' in this project`,
				404,
			);
		}

		// Return the first (and should be only) config
		return createSuccessResponse(configs[0]);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Get provider config API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}

// PUT /api/v1/providers/{providerName}/config - Update config for provider
export async function PUT(
	req: NextRequest,
	{ params }: { params: Promise<{ providerName: string }> },
) {
	try {
		const { providerName } = await params;
		const body = await req.json();
		const url = new URL(req.url);
		const projectId = url.searchParams.get("project_id");

		// Extract API key from headers
		const apiKey = extractApiKey(req);
		if (!apiKey) {
			return createAuthError();
		}

		if (!projectId) {
			return createErrorResponse("project_id parameter is required", 400);
		}

		// Validate request body
		const validationResult = updateConfigSchema.safeParse(body);
		if (!validationResult.success) {
			return createErrorResponse(
				"Invalid request format",
				400,
				validationResult.error.issues,
			);
		}

		// Find provider by name with caching
		const provider = await withCache(
			`provider:${providerName}:${projectId}`,
			() => api.providers.getByName({ name: providerName, projectId, apiKey }),
			300, // 5 minutes cache
		);

		if (!provider) {
			return createErrorResponse(`Provider '${providerName}' not found`, 404);
		}

		// Get existing config with caching
		const configs = await withCache(
			`provider-config:${projectId}:${provider.id}`,
			() =>
				api.providerConfigs.getAll({
					projectId,
					providerId: provider.id,
					apiKey,
				}),
			60, // 1 minute cache for user configs
		);

		const config = configs[0];
		if (!config) {
			return createErrorResponse(
				`No configuration found for provider '${providerName}' in this project`,
				404,
			);
		}

		// Update the config
		const updatedConfig = await api.providerConfigs.update({
			id: config.id,
			...validationResult.data,
			apiKey,
		});

		// Invalidate provider config cache
		await invalidateProviderConfigCache(projectId, provider.id);

		return createSuccessResponse(updatedConfig);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Update provider config API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}

// DELETE /api/v1/providers/{providerName}/config - Delete config for provider
export async function DELETE(
	req: NextRequest,
	{ params }: { params: Promise<{ providerName: string }> },
) {
	try {
		const { providerName } = await params;
		const url = new URL(req.url);
		const projectId = url.searchParams.get("project_id");

		// Extract API key from headers
		const apiKey = extractApiKey(req);
		if (!apiKey) {
			return createAuthError();
		}

		if (!projectId) {
			return createErrorResponse("project_id parameter is required", 400);
		}

		// Find provider by name with caching
		const provider = await withCache(
			`provider:${providerName}:${projectId}`,
			() => api.providers.getByName({ name: providerName, projectId, apiKey }),
			300, // 5 minutes cache
		);

		if (!provider) {
			return createErrorResponse(`Provider '${providerName}' not found`, 404);
		}

		// Get existing config with caching
		const configs = await withCache(
			`provider-config:${projectId}:${provider.id}`,
			() =>
				api.providerConfigs.getAll({
					projectId,
					providerId: provider.id,
					apiKey,
				}),
			60, // 1 minute cache for user configs
		);

		const config = configs[0];
		if (!config) {
			return createErrorResponse(
				`No configuration found for provider '${providerName}' in this project`,
				404,
			);
		}

		// Delete the config
		await api.providerConfigs.delete({
			id: config.id,
			apiKey,
		});

		// Invalidate provider config cache
		await invalidateProviderConfigCache(projectId, provider.id);

		return createSuccessResponse({ success: true });
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Delete provider config API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}
