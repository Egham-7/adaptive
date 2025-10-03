import type { CountTokensParameters } from "@google/genai";
import { GoogleGenAI } from "@google/genai";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { extractModelFromGeminiParam } from "@/lib/gemini-utils";
import { safeParseJson } from "@/lib/server/json-utils";
import { api } from "@/trpc/server";

export const dynamic = "force-dynamic";

export async function POST(
	req: NextRequest,
	{ params }: { params: { model: string } },
) {
	// Parse JSON with proper typing - let Gemini SDK handle validation
	const body = await safeParseJson<CountTokensParameters>(req);

	try {
		// Handle Gemini colon syntax: "model:countTokens" -> "model"
		const model = extractModelFromGeminiParam(params.model);

		// Extract API key from headers (Google uses x-goog-api-key primarily)
		const apiKey =
			req.headers.get("x-goog-api-key") ||
			req.headers.get("authorization")?.replace("Bearer ", "") ||
			req.headers.get("x-api-key") ||
			req.headers.get("api-key");

		if (!apiKey) {
			return new Response(
				JSON.stringify({
					error: {
						code: 401,
						message:
							"API key required. Provide it via x-goog-api-key, Authorization: Bearer, X-API-Key, or api-key header",
						status: "UNAUTHENTICATED",
					},
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Verify API key
		const verificationResult = await api.api_keys.verify({ apiKey });
		if (!verificationResult.valid) {
			return new Response(
				JSON.stringify({
					error: {
						code: 401,
						message: "Invalid API key",
						status: "UNAUTHENTICATED",
					},
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Use Google Gen AI SDK to call our backend with custom baseUrl
		const ai = new GoogleGenAI({
			apiKey: "internal", // Internal communication - will be handled by backend
			httpOptions: {
				baseUrl: env.ADAPTIVE_API_BASE_URL,
			},
		});

		// Build request parameters
		const countParams = {
			model,
			contents: body.contents,
		};

		// Call the countTokens method
		const result = await ai.models.countTokens(countParams);

		return new Response(JSON.stringify(result), {
			status: 200,
			headers: { "Content-Type": "application/json" },
		});
	} catch (error) {
		console.error("Gemini countTokens API error:", error);
		return new Response(
			JSON.stringify({
				error: {
					code: 500,
					message: "Internal server error",
					status: "INTERNAL",
				},
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}
