import Anthropic from "@anthropic-ai/sdk";
import type { NextRequest } from "next/server";
import { env } from "@/env";
import { api } from "@/trpc/server";
import { anthropicMessagesRequestSchema } from "@/types/anthropic-messages";
import { type ProviderType, providerEnum } from "@/types/providers";

export const runtime = "nodejs";

// POST /api/v1/clusters/{projectId}/{name}/messages - Anthropic-compatible messages with cluster routing
export async function POST(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string; name: string }> },
) {
	try {
		const { projectId, name } = await params;

		const rawBody = await request.json();
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
		const authHeader = request.headers.get("authorization");
		const bearerToken = authHeader?.startsWith("Bearer ")
			? authHeader.slice(7).replace(/\s+/g, "") || null
			: null;

		const apiKey =
			bearerToken ||
			request.headers.get("x-api-key") ||
			request.headers.get("api-key") ||
			request.headers.get("x-stainless-api-key");

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

		// Check that API key's project matches the requested projectId
		if (verificationResult.projectId !== projectId) {
			return new Response(
				JSON.stringify({
					type: "error",
					error: {
						type: "forbidden_error",
						message: "API key does not have access to the specified project",
					},
				}),
				{
					status: 403,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Get cluster by name with secure lookup (using API key's project for authorization)
		const cluster = await api.llmClusters.getByName({
			projectId: verificationResult.projectId,
			name,
		});

		if (!cluster) {
			return new Response(
				JSON.stringify({
					type: "error",
					error: {
						type: "not_found_error",
						message: `Cluster '${name}' not found in project '${projectId}'`,
					},
				}),
				{
					status: 404,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		// Use Anthropic SDK to call our backend with cluster routing
		const anthropic = new Anthropic({
			apiKey: "internal", // Internal communication - no real API key needed
			baseURL: `${env.ADAPTIVE_API_BASE_URL}/clusters/${projectId}/${name}`,
		});

		const startTime = Date.now();

		if (body.stream) {
			let finalUsage: Anthropic.MessageDeltaUsage | null = null;
			let modelName: string | null = null;
			let providerId: ProviderType | null = null;

			// Create custom ReadableStream that intercepts Anthropic SDK chunks
			const customReadable = new ReadableStream({
				async start(controller) {
					try {
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
							...(body.stop_sequences && {
								stop_sequences: body.stop_sequences,
							}),
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
								semantic_cache: body.semantic_cache,
							}),
							...(body.prompt_cache && { prompt_cache: body.prompt_cache }),
							...(body.fallback && { fallback: body.fallback }),
						} as Anthropic.MessageStreamParams);

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
								const rawProviderId = (
									messageStart.message as Anthropic.Message & {
										provider?: string;
									}
								).provider;
								providerId =
									rawProviderId &&
									providerEnum.includes(rawProviderId as ProviderType)
										? (rawProviderId as ProviderType)
										: null;
							}

							// Format as proper SSE with event type and data
							const eventType = chunk.type;
							const sseData = `event: ${eventType}\ndata: ${JSON.stringify(chunk)}\n\n`;
							controller.enqueue(new TextEncoder().encode(sseData));
						}

						// Send proper SSE termination event
						controller.enqueue(
							new TextEncoder().encode("event: done\ndata: [DONE]\n\n"),
						);
						controller.close();

						// Record usage if available from chunks
						if (finalUsage) {
							const usage = finalUsage; // Capture for closure
							queueMicrotask(async () => {
								try {
									await api.usage.recordApiUsage({
										apiKey,
										provider: providerId,
										model: modelName,
										usage: {
											promptTokens: usage.input_tokens ?? 0,
											completionTokens: usage.output_tokens ?? 0,
											totalTokens:
												(usage.input_tokens ?? 0) + (usage.output_tokens ?? 0),
										},
										duration: Date.now() - startTime,
										timestamp: new Date(),
										clusterId: cluster.id,
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
									provider: "anthropic" as ProviderType,
									model: null,
									usage: {
										promptTokens: 0,
										completionTokens: 0,
										totalTokens: 0,
									},
									duration: Date.now() - startTime,
									timestamp: new Date(),
									clusterId: cluster.id,
								});
							} catch (usageError) {
								console.error("Failed to record error usage:", usageError);
							}
						});
					}
				},
			});

			return new Response(customReadable, {
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				},
			});
		}
		// Handle non-streaming
		const message = (await anthropic.messages.create({
			model: body.model,
			max_tokens: body.max_tokens,
			messages: body.messages,
			...(body.system && { system: body.system }),
			...(body.temperature !== undefined && { temperature: body.temperature }),
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
			...(body.semantic_cache && { semantic_cache: body.semantic_cache }),
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
						clusterId: cluster.id,
					});
				} catch (error) {
					console.error("Failed to record usage:", error);
				}
			});
		}

		return Response.json(message);
	} catch (error) {
		console.error("Cluster Anthropic Messages API error:", error);
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
