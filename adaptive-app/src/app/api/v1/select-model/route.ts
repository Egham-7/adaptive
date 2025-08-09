import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import type { NextRequest } from "next/server";
import { api } from "@/trpc/server";
import { selectModelRequestSchema } from "@/types/select-model";

export async function POST(req: NextRequest) {
	try {
		let rawBody: unknown;
		try {
			rawBody = await req.json();
		} catch (jsonError) {
			// Handle JSON parsing errors specifically
			if (jsonError instanceof SyntaxError) {
				return new Response(
					JSON.stringify({
						error: {
							message: "Invalid JSON input",
							details: "Request body must be valid JSON",
						},
					}),
					{
						status: 400,
						headers: { "Content-Type": "application/json" },
					},
				);
			}
			throw jsonError; // Re-throw non-JSON errors
		}

		// Validate request body against schema
		const validationResult = selectModelRequestSchema.safeParse(rawBody);
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

		const body = validationResult.data;

		// Extract API key from OpenAI-compatible headers
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

		// Call the tRPC selectModel procedure
		const result = await api.selectModel.selectModel({
			apiKey,
			request: body,
		});

		return Response.json(result);
	} catch (cause) {
		// Handle tRPC errors properly
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

		// Handle non-tRPC errors
		console.error("Select model API error:", cause);
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
