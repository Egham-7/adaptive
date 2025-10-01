import Anthropic from "@anthropic-ai/sdk";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { safeParseJson } from "@/lib/server/json-utils";
import { api } from "@/trpc/server";
import {
	type AdaptiveAnthropicResponse,
	anthropicMessagesRequestSchema,
} from "@/types/anthropic-messages";
import type { ProviderType } from "@/types/providers";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
	const rawBody = await safeParseJson(req);

	try {
		const validationResult = anthropicMessagesRequestSchema.safeParse(rawBody);

		if (!validationResult.success) {
			return new Response(
				JSON.stringify({
					type: "error",
					error: {
						type: "validation_error",
						message: "Invalid request body",
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
					type: "error",
					error: {
						type: "authentication_error",
						message:
							"API key required. Provide it via Authorization: Bearer, X-API-Key, api-key, or X-Stainless-API-Key header",
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
					type: "error",
					error: {
						type: "authentication_error",
						message: "Invalid API key",
					},
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Use Anthropic SDK to call our backend
		const anthropic = new Anthropic({
			apiKey: "internal", // Internal communication - no real API key needed
			baseURL: env.ADAPTIVE_API_BASE_URL,
		});

		const startTime = Date.now();

		console.log("Stream:", body.stream);

		if (body.stream) {
			const encoder = new TextEncoder(); // Reuse encoder

			// Create custom ReadableStream that intercepts Anthropic SDK chunks
			const customReadable = new ReadableStream({
				async start(controller) {
					let finalUsage: Anthropic.MessageDeltaUsage | null = null; // Minimize scope
					let modelName: string | null = null;
					let _providerId: string | null = null;

					try {
						// Build validated stream parameters - use base Anthropic types to avoid conflicts
						const baseParams = {
							model: body.model,
							max_tokens: body.max_tokens,
							messages: body.messages,
							...(body.system && { system: body.system }),
							...(body.temperature !== undefined && {
								temperature: body.temperature,
							}),
							...(body.top_p !== undefined && { top_p: body.top_p }),
							...(body.top_k !== undefined && { top_k: body.top_k }),
							...(body.stop_sequences && {
								stop_sequences: body.stop_sequences,
							}),
							...(body.metadata && { metadata: body.metadata }),
							...(body.tools && { tools: body.tools }),
							...(body.tool_choice && { tool_choice: body.tool_choice }),
						};

						// Add custom extensions separately to avoid type conflicts
						const streamParams = {
							...baseParams,
							// Custom adaptive extensions
							...(body.provider_configs && {
								provider_configs: body.provider_configs,
							}),
							...(body.model_router && {
								model_router: body.model_router,
							}),
							...(body.semantic_cache && {
								prompt_response_cache: {
									enabled: body.semantic_cache.enabled,
									semantic_threshold: body.semantic_cache.semantic_threshold,
								},
							}),
							...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
							...(body.fallback && { fallback: body.fallback }),
						};

						const stream = anthropic.messages.stream(
							streamParams as Anthropic.MessageStreamParams,
						);

						for await (const chunk of stream) {
							// Track usage data from message_delta chunks
							if (chunk.type === "message_delta") {
								const messageDelta = chunk as Anthropic.MessageDeltaEvent;
								if (messageDelta.usage) {
									finalUsage = messageDelta.usage;
								}
							}

							// Track message_start for model and provider info
							if (chunk.type === "message_start") {
								const messageStart = chunk as Anthropic.MessageStartEvent;
								modelName = messageStart.message.model || null;
								// Provider info might be in extended response
								_providerId =
									(
										messageStart.message as Anthropic.Message & {
											provider?: string;
										}
									).provider || null;
							}

							// Format as proper SSE with event type and data
							const eventType = chunk.type;
							const sseData = `event: ${eventType}\ndata: ${JSON.stringify(chunk)}\n\n`;
							controller.enqueue(encoder.encode(sseData));
						}

						// Send proper SSE termination event
						controller.enqueue(encoder.encode("event: done\ndata: [DONE]\n\n"));
						controller.close();

						// Record usage if available from chunks
						if (finalUsage) {
							const usage = finalUsage; // Capture for closure
							queueMicrotask(async () => {
								try {
									await api.usage.recordApiUsage({
										apiKey,
										provider: "anthropic" as ProviderType,
										model: modelName,
										usage: {
											promptTokens: usage.input_tokens ?? 0,
											completionTokens: usage.output_tokens ?? 0,
											totalTokens:
												(usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
										},
										duration: Date.now() - startTime,
										timestamp: new Date(),
									});
								} catch (error) {
									console.error("Failed to record streaming usage:", error);
								}
							});
						}
					} catch (error) {
						console.error("Streaming error:", error);
						const errorData = `event: error\ndata: ${JSON.stringify({
							type: "error",
							error: { message: "Stream failed", type: "stream_error" },
						})}\n\n`;
						controller.enqueue(encoder.encode(errorData));
						controller.close();

						// Record error usage
						queueMicrotask(async () => {
							try {
								await api.usage.recordApiUsage({
									apiKey,
									provider: "anthropic" as ProviderType,
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
		}
		// Handle non-streaming with Anthropic SDK - use same safe approach
		const baseParams = {
			model: body.model,
			max_tokens: body.max_tokens,
			messages: body.messages,
			...(body.system && { system: body.system }),
			...(body.temperature !== undefined && {
				temperature: body.temperature,
			}),
			...(body.top_p !== undefined && { top_p: body.top_p }),
			...(body.top_k !== undefined && { top_k: body.top_k }),
			...(body.stop_sequences && { stop_sequences: body.stop_sequences }),
			...(body.metadata && { metadata: body.metadata }),
			...(body.tools && { tools: body.tools }),
			...(body.tool_choice && { tool_choice: body.tool_choice }),
		};

		const createParams = {
			...baseParams,
			// Custom adaptive extensions
			...(body.provider_configs && {
				provider_configs: body.provider_configs,
			}),
			...(body.model_router && {
				model_router: body.model_router,
			}),
			...(body.semantic_cache && {
				prompt_response_cache: {
					enabled: body.semantic_cache.enabled,
					semantic_threshold: body.semantic_cache.semantic_threshold,
				},
			}),
			...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
			...(body.fallback && { fallback: body.fallback }),
		};

		const message = (await anthropic.messages.create(
			createParams as Anthropic.MessageCreateParams,
		)) as AdaptiveAnthropicResponse;

		// Record usage
		if (message.usage) {
			const usage = message.usage; // Capture for closure
			queueMicrotask(async () => {
				try {
					await api.usage.recordApiUsage({
						apiKey,
						provider: "anthropic" as const,
						model: message.model,
						usage: {
							promptTokens: usage.input_tokens,
							completionTokens: usage.output_tokens,
							totalTokens: usage.input_tokens + usage.output_tokens,
						},
						duration: Date.now() - startTime,
						timestamp: new Date(),
					});
				} catch (error) {
					console.error("Failed to record usage:", error);
				}
			});
		}

		return Response.json(message);
	} catch (error) {
		console.error("Anthropic Messages API error:", error);
		return new Response(
			JSON.stringify({
				type: "error",
				error: {
					type: "api_error",
					message: "Internal server error",
				},
			}),
			{
				status: 500,
				headers: { "Content-Type": "application/json" },
			},
		);
	}
}
