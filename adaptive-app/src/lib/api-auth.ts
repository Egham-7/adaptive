import type { NextRequest } from "next/server";

/**
 * Extracts API key from OpenAI-compatible request headers
 */
export function extractApiKey(req: NextRequest): string | null {
	const authHeader = req.headers.get("authorization");
	const bearerToken = authHeader?.startsWith("Bearer ")
		? authHeader.slice(7).replace(/\s+/g, "") || null
		: null;

	return (
		bearerToken ||
		req.headers.get("x-api-key") ||
		req.headers.get("api-key") ||
		req.headers.get("x-stainless-api-key")
	);
}

/**
 * Creates standardized 401 Unauthorized response for missing API key
 */
export function createAuthError(): Response {
	return new Response(
		JSON.stringify({
			error: {
				message: "API key required",
				details:
					"Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
			},
		}),
		{
			status: 401,
			headers: { "Content-Type": "application/json" },
		},
	);
}

/**
 * Creates standardized error response
 */
export function createErrorResponse(
	message: string,
	status: number,
	details?: unknown,
): Response {
	const errorBody: { message: string; details?: unknown } = { message };
	if (details) {
		errorBody.details = details;
	}

	return new Response(JSON.stringify({ error: errorBody }), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}

/**
 * Creates standardized success response
 */
export function createSuccessResponse(data: unknown, status = 200): Response {
	return Response.json(data, { status });
}
