import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import type { NextRequest } from "next/server";
import { api } from "@/trpc/server";
import type { SelectModelRequest } from "@/types/select-model";

export async function POST(req: NextRequest) {
	try {
		const body: SelectModelRequest = await req.json();

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
