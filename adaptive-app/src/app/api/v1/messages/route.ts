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
			// Handle streaming by calling our backend directly
			// The Anthropic SDK has issues when pointing to custom backends
			try {
				const backendUrl = `${env.ADAPTIVE_API_BASE_URL}/v1/messages`;

				const response = await fetch(backendUrl, {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${jwtToken}`,
					},
					body: JSON.stringify({
						model: body.model,
						max_tokens: body.max_tokens,
						messages: body.messages,
						stream: true,
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
					}),
				});

				if (!response.ok) {
					throw new Error(
						`Backend responded with ${response.status}: ${response.statusText}`,
					);
				}

				if (!response.body) {
					throw new Error("No response body received from backend");
				}

				// Create a ReadableStream that forwards the backend stream
				const readableStream = new ReadableStream({
					async start(controller) {
						const reader = response.body?.getReader();
						const decoder = new TextDecoder();
						let finalMessage: any = null;

						try {
							while (true) {
								const { done, value } = await reader.read();

								if (done) {
									console.log("Backend stream completed");
									break;
								}

								// Decode the chunk and forward it
								const chunk = decoder.decode(value, { stream: true });
								controller.enqueue(new TextEncoder().encode(chunk));

								// Try to parse usage information for recording
								const lines = chunk.split("\n");
								for (const line of lines) {
									if (line.startsWith("data: ") && !line.includes("[DONE]")) {
										try {
											const data = JSON.parse(line.slice(6));
											if (
												data.type === "message_delta" &&
												data.delta?.stop_reason === "end_turn"
											) {
												// Extract message info for usage recording
												finalMessage = data.message;
											}
										} catch (_e) {
											// Ignore parsing errors
										}
									}
								}
							}
						} catch (error) {
							console.error("Stream reading error:", error);
							const errorData = `event: error\ndata: ${JSON.stringify({
								type: "error",
								error: {
									message:
										error instanceof Error ? error.message : "Stream error",
									type: "stream_error",
								},
							})}\n\n`;
							controller.enqueue(new TextEncoder().encode(errorData));
						} finally {
							controller.close();

							// Record usage if available
							if (finalMessage?.usage) {
								queueMicrotask(async () => {
									try {
										await api.usage.recordApiUsage({
											apiKey,
											provider: "anthropic",
											model: finalMessage.model ?? null,
											usage: {
												promptTokens: finalMessage.usage.input_tokens ?? 0,
												completionTokens: finalMessage.usage.output_tokens ?? 0,
												totalTokens:
													(finalMessage.usage.input_tokens ?? 0) +
													(finalMessage.usage.output_tokens ?? 0),
											},
											duration: Date.now() - startTime,
											timestamp: new Date(),
										});
									} catch (error) {
										console.error("Failed to record usage:", error);
									}
								});
							}
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
			} catch (error) {
				console.error("Streaming error:", error);

				const errorMessage =
					error instanceof Error ? error.message : "Stream failed";
				const errorData = `event: error\ndata: ${JSON.stringify({
					type: "error",
					error: {
						message: errorMessage,
						type: "stream_error",
					},
				})}\n\n`;

				const errorStream = new ReadableStream({
					start(controller) {
						controller.enqueue(new TextEncoder().encode(errorData));
						controller.close();
					},
				});

				return new Response(errorStream, {
					headers: {
						"Content-Type": "text/event-stream",
						"Cache-Control": "no-cache",
						Connection: "keep-alive",
					},
				});
			}
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
