import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import type { NextRequest } from "next/server";
import {
	createAuthError,
	createErrorResponse,
	createSuccessResponse,
	extractApiKey,
} from "@/lib/auth/clerk";
import { invalidateProviderCache, withCache } from "@/lib/shared/cache";
import { api } from "@/trpc/server";
import { createProviderSchema } from "@/types/providers";

// GET /api/v1/providers - List providers with user configs
export async function GET(req: NextRequest) {
	try {
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

		// Get providers and configs in parallel with caching
		const [providers, configs] = await Promise.all([
			withCache(
				`providers:${projectId}`,
				() => api.providers.getAll({ apiKey, projectId }),
				300, // 5 minutes cache
			),
			withCache(
				`provider-configs:${projectId}`,
				() => api.providerConfigs.getAll({ apiKey, projectId }),
				60, // 1 minute cache (shorter for user configs)
			),
		]);

		// Create a map of configs by provider ID for quick lookup
		const configMap = new Map(
			configs.map((config) => [config.providerId, config]),
		);

		// Merge providers with their configs
		const providersWithConfigs = providers.map((provider) => ({
			...provider,
			config: configMap.get(provider.id) || null,
		}));

		return createSuccessResponse({
			providers: providersWithConfigs,
			count: providersWithConfigs.length,
		});
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Get providers API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}

// POST /api/v1/providers - Create a new provider template
export async function POST(req: NextRequest) {
	try {
		const body = await req.json();

		// Extract API key from headers
		const apiKey = extractApiKey(req);
		if (!apiKey) {
			return createAuthError();
		}

		// Validate request body
		const validationResult = createProviderSchema.safeParse(body);
		if (!validationResult.success) {
			return createErrorResponse(
				"Invalid request format",
				400,
				validationResult.error.issues,
			);
		}

		// Create provider template using tRPC
		const provider = await api.providers.create({
			...validationResult.data,
			apiKey,
		});

		// Invalidate provider cache for all projects (since this is a system-wide provider)
		await invalidateProviderCache("*");

		return createSuccessResponse(provider, 201);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return createErrorResponse(cause.message, httpStatusCode);
		}

		console.error("Create provider API error:", cause);
		return createErrorResponse("Internal server error", 500);
	}
}
