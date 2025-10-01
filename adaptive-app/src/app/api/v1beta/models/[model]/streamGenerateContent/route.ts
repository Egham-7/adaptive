import { GoogleGenAI } from "@google/genai";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { extractModelFromGeminiParam } from "@/lib/gemini-utils";
import { safeParseJson } from "@/lib/server/json-utils";
import { api } from "@/trpc/server";
import type {
	AdaptiveGeminiChunk,
	AdaptiveGeminiRequest,
	AdaptiveGeminiUsage,
} from "@/types/gemini-generate";
import type { ProviderType } from "@/types/providers";

export const dynamic = "force-dynamic";

export async function POST(
	req: NextRequest,
	{ params }: { params: { model: string } },
) {
	// Parse JSON with proper typing - let Gemini SDK handle validation
	const body = await safeParseJson<AdaptiveGeminiRequest>(req);

	try {
		// Handle Gemini colon syntax: "model:streamGenerateContent" -> "model"
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
		const encoder = new TextEncoder();

		// Create custom ReadableStream that handles Gemini streaming format
		const customReadable = new ReadableStream({
			async start(controller) {
				let finalUsage: AdaptiveGeminiUsage | null = null;
				let modelName: string | null = null;
				let providerId: string | null = null;

				try {
					// Build request parameters following the SDK pattern
					const streamParams = {
						model, // The model name from the URL parameter
						contents: body.contents,
						...(body.config && { config: body.config }),
						// Add Adaptive-specific extensions
						...(body.provider_configs && {
							provider_configs: body.provider_configs,
						}),
						...(body.model_router && { model_router: body.model_router }),
						...(body.semantic_cache && { semantic_cache: body.semantic_cache }),
						...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
						...(body.fallback && { fallback: body.fallback }),
					};

					// Use the models.generateContentStream method as shown in SDK docs
					const streamResult =
						await ai.models.generateContentStream(streamParams);

					// Stream through the async generator
					for await (const chunk of streamResult) {
						const typedChunk = chunk as AdaptiveGeminiChunk;

						// Track usage data from chunks
						if (typedChunk.usageMetadata) {
							finalUsage = typedChunk.usageMetadata;
						}

						// Track model and provider info - these will always be present
						if (typedChunk.modelVersion) {
							modelName = typedChunk.modelVersion;
						}
						providerId = typedChunk.provider;

						// Format as Server-Sent Event following Gemini's format
						// Gemini uses data-only format, not event-prefixed like others
						const sseData = `data: ${JSON.stringify(typedChunk)}\n\n`;
						controller.enqueue(encoder.encode(sseData));
					}

					controller.close();

					// Record usage if available from final chunk
					if (finalUsage) {
						const usage = finalUsage; // Capture for closure
						queueMicrotask(async () => {
							try {
								await api.usage.recordApiUsage({
									apiKey,
									provider: (providerId ?? "gemini") as ProviderType,
									model: modelName ?? model,
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
								console.error("Failed to record streaming usage:", error);
							}
						});
					}
				} catch (error) {
					console.error("Streaming error:", error);
					const errorData = `data: ${JSON.stringify({
						error: {
							code: 500,
							message: "Stream failed",
							status: "INTERNAL",
						},
					})}\n\n`;
					controller.enqueue(encoder.encode(errorData));
					controller.close();

					// Record error usage
					queueMicrotask(async () => {
						try {
							await api.usage.recordApiUsage({
								apiKey,
								provider: "gemini",
								model: null,
								usage: {
									promptTokens: 0,
									completionTokens: 0,
									totalTokens: 0,
								},
								duration: Date.now() - startTime,
								timestamp: new Date(),
								requestCount: 1,
								error: error instanceof Error ? error.message : String(error),
							});
						} catch (usageError) {
							console.error("Failed to record streaming error:", usageError);
						}
					});
				}
			},
		});

		return new Response(customReadable, {
			headers: {
				"Content-Type": "text/event-stream; charset=utf-8",
				"Cache-Control": "no-cache, no-transform",
				Pragma: "no-cache",
				"X-Accel-Buffering": "no",
				Connection: "keep-alive",
			},
		});
	} catch (error) {
		console.error("Gemini streamGenerateContent API error:", error);
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
