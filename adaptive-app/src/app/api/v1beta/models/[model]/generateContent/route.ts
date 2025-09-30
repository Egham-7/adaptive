import { GoogleGenAI } from "@google/genai";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { extractModelFromGeminiParam } from "@/lib/gemini-utils";
import { safeParseJson } from "@/lib/server/json-utils";
import { api } from "@/trpc/server";
import type {
	AdaptiveGeminiRequest,
	AdaptiveGeminiResponse,
	AdaptiveGeminiUsage,
} from "@/types/gemini-generate";

export const dynamic = "force-dynamic";

export async function POST(
	req: NextRequest,
	{ params }: { params: { model: string } },
) {
	// Parse JSON with proper typing - let Gemini SDK handle validation
	const body = await safeParseJson<AdaptiveGeminiRequest>(req);

	try {
		// Handle Gemini colon syntax: "model:generateContent" -> "model"
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

		const startTime = Date.now();

		// Build request parameters following the SDK pattern
		const generateParams = {
			model, // The model name from the URL parameter
			contents: body.contents,
			...(body.config && { config: body.config }),
			// Add Adaptive-specific extensions
			...(body.provider_configs && { provider_configs: body.provider_configs }),
			...(body.model_router && { model_router: body.model_router }),
			...(body.semantic_cache && { semantic_cache: body.semantic_cache }),
			...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
			...(body.fallback && { fallback: body.fallback }),
		};

		// Use the models.generateContent method as shown in SDK docs
		const response = (await ai.models.generateContent(
			generateParams,
		)) as AdaptiveGeminiResponse;

		// Record usage if available
		if (response.usageMetadata) {
			const usage = response.usageMetadata as AdaptiveGeminiUsage;
			queueMicrotask(async () => {
				try {
					await api.usage.recordApiUsage({
						apiKey,
						provider: response.provider as
							| "openai"
							| "anthropic"
							| "gemini"
							| "groq"
							| "deepseek"
							| "huggingface"
							| "grok"
							| "adaptive",
						model: response.modelVersion ?? model,
						usage: {
							promptTokens: usage.promptTokenCount ?? 0,
							completionTokens: usage.candidatesTokenCount ?? 0,
							totalTokens: usage.totalTokenCount ?? 0,
						},
						duration: Date.now() - startTime,
						timestamp: new Date(),
						cacheTier: usage.cache_tier,
					});
				} catch (error) {
					console.error("Failed to record usage:", error);
				}
			});
		}

		return Response.json(response);
	} catch (error) {
		console.error("Gemini generateContent API error:", error);
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
