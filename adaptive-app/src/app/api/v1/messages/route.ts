import Anthropic from "@anthropic-ai/sdk";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { createBackendJWT } from "@/lib/jwt";
import { api } from "@/trpc/server";
import { anthropicMessagesRequestSchema } from "@/types/anthropic-messages";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
	try {
		const rawBody = await req.json();
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

		// Create JWT token for backend authentication
		const jwtToken = await createBackendJWT(apiKey);

		// Use Anthropic SDK to call our backend
		const anthropic = new Anthropic({
			apiKey: jwtToken, // Use JWT token instead of API key
			baseURL: env.ADAPTIVE_API_BASE_URL,
		});

		const startTime = Date.now();

		console.log("Stream:", body.stream);

		if (body.stream) {
			// Handle streaming with Anthropic SDK using .on() event handlers for type safety
			const stream = anthropic.messages.stream({
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
				// Custom adaptive extensions (cast to bypass SDK type checking)
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
			} as Anthropic.MessageStreamParams);

			// Create ReadableStream using Anthropic SDK's high-level event handlers
			const readableStream = new ReadableStream({
				async start(controller) {
					let finalMessage: Anthropic.Message | null = null;

					try {
						// Use Anthropic SDK's cleaner high-level event handlers
						stream
							.on("connect", () => {
								console.log("Connected to Anthropic API");
							})
							.on(
								"streamEvent",
								(
									event: Anthropic.MessageStreamEvent,
									_snapshot: Anthropic.Message,
								) => {
									// Send the raw SSE event for compatibility with existing frontend
									const sseData = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
									controller.enqueue(new TextEncoder().encode(sseData));
								},
							)
							.on("text", (textDelta: string, _textSnapshot: string) => {
								// Optional: Could use this for text-only streaming optimizations
								console.log("Text delta:", textDelta);
							})
							.on("message", (message: Anthropic.Message) => {
								console.log("Message completed:", message.id);
								finalMessage = message;
							})
							.on("finalMessage", (message: Anthropic.Message) => {
								console.log("Final message received:", message.id);
								finalMessage = message;

								// Send termination event
								controller.enqueue(
									new TextEncoder().encode("event: done\ndata: [DONE]\n\n"),
								);
								controller.close();
							})
							.on("error", (error: Error) => {
								console.error("Anthropic stream error:", error);
								const errorData = `event: error\ndata: ${JSON.stringify({
									type: "error",
									error: {
										message: error.message || "Stream error",
										type: "stream_error",
									},
								})}\n\n`;
								controller.enqueue(new TextEncoder().encode(errorData));
								controller.close();
							})
							.on("abort", (error: Error) => {
								console.log("Stream aborted:", error.message);
								controller.close();
							})
							.on("end", () => {
								console.log("Stream ended");
								// Note: ReadableStreamDefaultController doesn't have a 'closed' property
								// We'll rely on the finalMessage event to close the controller
							});

						// Wait for stream completion
						await stream.done();

						// Record usage if available (finalMessage is set in the event handlers above)
						const messageForUsage = finalMessage as Anthropic.Message | null;
						if (messageForUsage?.usage) {
							const messageUsage = messageForUsage.usage;
							const messageModel = messageForUsage.model;
							queueMicrotask(async () => {
								try {
									await api.usage.recordApiUsage({
										apiKey,
										provider: "anthropic", // Default since we're using Anthropic format
										model: messageModel ?? null,
										usage: {
											promptTokens: messageUsage.input_tokens ?? 0,
											completionTokens: messageUsage.output_tokens ?? 0,
											totalTokens:
												(messageUsage.input_tokens ?? 0) +
												(messageUsage.output_tokens ?? 0),
										},
										duration: Date.now() - startTime,
										timestamp: new Date(),
									});
								} catch (error) {
									console.error("Failed to record usage:", error);
								}
							});
						}
					} catch (error) {
						console.error("Stream error:", error);

						// Extract meaningful error message
						let errorMessage = "Stream failed";
						let errorType = "stream_error";

						if (error instanceof Error) {
							if (error.message.includes("API key not configured")) {
								errorMessage = error.message;
								errorType = "configuration_error";
							} else if (
								error.message.includes("request ended without sending")
							) {
								errorMessage = "Request timeout. Please try again.";
								errorType = "timeout_error";
							} else {
								errorMessage = error.message;
							}
						}

						// Send proper SSE error event before closing
						const errorData = `event: error\ndata: ${JSON.stringify({
							type: "error",
							error: {
								message: errorMessage,
								type: errorType,
							},
						})}\n\n`;
						controller.enqueue(new TextEncoder().encode(errorData));
						controller.close();

						// Record error usage
						queueMicrotask(async () => {
							try {
								await api.usage.recordApiUsage({
									apiKey,
									provider: "anthropic",
									model: null,
									usage: {
										promptTokens: 0,
										completionTokens: 0,
										totalTokens: 0,
									},
									duration: Date.now() - startTime,
									timestamp: new Date(),
								});
							} catch (usageError) {
								console.error("Failed to record error usage:", usageError);
							}
						});
					}
				},
			});

			return new Response(readableStream, {
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				},
			});
		}
		// Handle non-streaming with Anthropic SDK
		const message = (await anthropic.messages.create({
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
			// Custom adaptive extensions (cast to bypass SDK type checking)
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
		} as Anthropic.MessageCreateParams)) as Anthropic.Message;

		// Record usage
		if (message.usage) {
			queueMicrotask(async () => {
				try {
					await api.usage.recordApiUsage({
						apiKey,
						provider: "anthropic", // Default since we're using Anthropic format
						model: message.model,
						usage: {
							promptTokens: message.usage.input_tokens,
							completionTokens: message.usage.output_tokens,
							totalTokens:
								message.usage.input_tokens + message.usage.output_tokens,
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
