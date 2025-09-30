import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import type { NextRequest } from "next/server";
import { safeParseJson } from "@/lib/server/json-utils";
import { api } from "@/trpc/server";
import {
	type UpdateProviderInput,
	updateProviderSchema,
} from "@/types/providers";

// GET /api/v1/providers/{providerId} - Get a specific provider
export async function GET(
	req: NextRequest,
	{ params }: { params: Promise<{ providerId: string }> },
) {
	try {
		const { providerId } = await params;

		// Extract API key from headers
		const authHeader = req.headers.get("authorization");
		const bearerToken = authHeader?.startsWith("Bearer ")
			? authHeader.slice(7).replace(/\s+/g, "") || null
			: null;

		const apiKey =
			bearerToken ||
			req.headers.get("x-api-key") ||
			req.headers.get("api-key") ||
			req.headers.get("x-stainless-api-key");

		if (!apiKey) {
			return new Response(
				JSON.stringify({
					error:
						"API key required. Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Get provider by ID using tRPC
		const provider = await api.providers.getById({
			id: providerId,
			apiKey,
		});

		return Response.json(provider);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return new Response(
				JSON.stringify({
					error: { message: cause.message },
				}),
				{
					status: httpStatusCode,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		console.error("Get provider API error:", cause);
		return new Response(
			JSON.stringify({
				error: { message: "Internal server error" },
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}

// PUT /api/v1/providers/{providerId} - Update a provider
export async function PUT(
	req: NextRequest,
	{ params }: { params: Promise<{ providerId: string }> },
) {
	const { providerId } = await params;
	const body = await safeParseJson<Omit<UpdateProviderInput, "id">>(req);

	try {
		// Extract API key from headers
		const authHeader = req.headers.get("authorization");
		const bearerToken = authHeader?.startsWith("Bearer ")
			? authHeader.slice(7).replace(/\s+/g, "") || null
			: null;

		const apiKey =
			bearerToken ||
			req.headers.get("x-api-key") ||
			req.headers.get("api-key") ||
			req.headers.get("x-stainless-api-key");

		if (!apiKey) {
			return new Response(
				JSON.stringify({
					error:
						"API key required. Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Validate request body
		const validationResult = updateProviderSchema.safeParse({
			...body,
			id: providerId,
		});

		if (!validationResult.success) {
			return new Response(
				JSON.stringify({
					error: {
						message: "Invalid request format",
						details: validationResult.error.issues,
					},
				}),
				{
					status: 400,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Update provider using tRPC
		const provider = await api.providers.update({
			...validationResult.data,
			apiKey,
		});

		return Response.json(provider);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return new Response(
				JSON.stringify({
					error: { message: cause.message },
				}),
				{
					status: httpStatusCode,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		console.error("Update provider API error:", cause);
		return new Response(
			JSON.stringify({
				error: { message: "Internal server error" },
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}

// DELETE /api/v1/providers/{providerId} - Delete a provider
export async function DELETE(
	req: NextRequest,
	{ params }: { params: Promise<{ providerId: string }> },
) {
	try {
		const { providerId } = await params;

		// Extract API key from headers
		const authHeader = req.headers.get("authorization");
		const bearerToken = authHeader?.startsWith("Bearer ")
			? authHeader.slice(7).replace(/\s+/g, "") || null
			: null;

		const apiKey =
			bearerToken ||
			req.headers.get("x-api-key") ||
			req.headers.get("api-key") ||
			req.headers.get("x-stainless-api-key");

		if (!apiKey) {
			return new Response(
				JSON.stringify({
					error:
						"API key required. Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Delete provider using tRPC
		const result = await api.providers.delete({
			id: providerId,
			apiKey,
		});

		return Response.json(result);
	} catch (cause) {
		if (cause instanceof TRPCError) {
			const httpStatusCode = getHTTPStatusCodeFromError(cause);
			return new Response(
				JSON.stringify({
					error: { message: cause.message },
				}),
				{
					status: httpStatusCode,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		console.error("Delete provider API error:", cause);
		return new Response(
			JSON.stringify({
				error: { message: "Internal server error" },
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}
