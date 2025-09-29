import type { NextRequest } from "next/server";
import OpenAI from "openai";
import { decryptProviderApiKey } from "@/lib/auth";
import { safeParseJson } from "@/lib/server/json-utils";
import {
	filterUsageFromChunk,
	userRequestedUsage,
	withUsageTracking,
} from "@/lib/server/usage-utils";
import { api } from "@/trpc/server";
import type {
	AuthType,
	ChatCompletion,
	ChatCompletionChunk,
	ChatCompletionRequest,
	ProviderConfig,
} from "@/types";
import { chatCompletionRequestSchema } from "@/types/chat-completion";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
	// Parse and validate request body outside try-catch to let validation errors bubble up naturally
	const rawBody = await safeParseJson(req);
	const validationResult = chatCompletionRequestSchema.safeParse(rawBody);

	if (!validationResult.success) {
		return new Response(
			JSON.stringify({
				error: "Invalid request body",
				details: validationResult.error.issues,
			}),
			{
				status: 400,
				headers: { "Content-Type": "application/json" },
			},
		);
	}

	const body = validationResult.data as ChatCompletionRequest;

	try {
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

		const verificationResult = await api.api_keys.verify({
			apiKey,
		});

		if (!verificationResult.valid) {
			return new Response(JSON.stringify({ error: "Invalid API key" }), {
				status: 401,
				headers: { "Content-Type": "application/json" },
			});
		}

		const { projectId } = verificationResult;

		// Fetch provider configurations from database if project is specified
		const providerConfigs: Record<string, ProviderConfig> = {};

		if (projectId) {
			try {
				const configs = await api.providerConfigs.getAll({
					projectId,
					apiKey,
				});

				// Transform database provider configs to Go backend format
				configs.forEach((config) => {
					const provider = config.provider;
					providerConfigs[provider.name] = {
						base_url: provider.baseUrl ?? undefined,
						auth_type: (provider.authType as AuthType | undefined) ?? undefined,
						auth_header_name: provider.authHeaderName ?? undefined,
						api_key: decryptProviderApiKey(config.providerApiKey), // Decrypt user's API key from config
						health_endpoint: provider.healthEndpoint ?? undefined,
						rate_limit_rpm: provider.rateLimitRpm ?? undefined,
						timeout_ms: provider.timeoutMs ?? undefined,
						retry_config:
							(provider.retryConfig as Record<string, unknown>) ?? undefined,
						headers: {
							...(provider.headers as Record<string, string>),
							...(config.customHeaders as Record<string, string>),
						},
					};
				});
			} catch (error) {
				console.warn("Failed to fetch provider configs:", error);
				// Continue without provider configs - will use default providers
			}
		}

		// Support both streaming and non-streaming requests
		const shouldStream = body.stream === true;

		// Check if user requested usage data
		const userWantsUsage = userRequestedUsage(body);
		// Only add include_usage: true for streaming requests
		// Non-streaming requests will get usage info by default from most providers
		const internalBody = shouldStream ? withUsageTracking(body) : body;

		const baseURL = `${process.env.ADAPTIVE_API_BASE_URL}/v1`;

		const openai = new OpenAI({
			apiKey: "internal", // Internal communication - no real API key needed
			baseURL,
		});

		if (shouldStream) {
			const streamStartTime = Date.now();
			const encoder = new TextEncoder(); // Reuse encoder
			const abortController = new AbortController();
			const timeoutMs = 300000; // 5 minutes timeout

			// Set up stream timeout
			const timeoutId = setTimeout(() => {
				abortController.abort();
			}, timeoutMs);

			// Create custom ReadableStream that intercepts OpenAI SDK chunks
			const customReadable = new ReadableStream({
				async start(controller) {
					try {
						const stream = await openai.chat.completions.create(
							{
								...internalBody,
								stream: true,
							},
							{
								body: {
									...internalBody,
									stream: true,
									provider_configs: providerConfigs,
								},
								signal: abortController.signal,
							},
						);

						for await (const chunk of stream) {
							const typedChunk = chunk as ChatCompletionChunk;

							// Record usage in background when we get it
							if (typedChunk.usage) {
								console.log("Usage: ", typedChunk.usage);
								queueMicrotask(async () => {
									try {
										await api.usage.recordApiUsage({
											apiKey,
											provider: typedChunk.provider ?? null,
											model: typedChunk.model ?? null,
											usage: {
												promptTokens: typedChunk.usage?.prompt_tokens ?? 0,
												completionTokens:
													typedChunk.usage?.completion_tokens ?? 0,
												totalTokens: typedChunk.usage?.total_tokens ?? 0,
											},
											duration: Date.now() - streamStartTime,
											timestamp: new Date(),
											cacheTier: typedChunk.usage?.cache_tier,
										});
									} catch (error) {
										console.error("Failed to record streaming usage:", error);
									}
								});
							}

							// Filter out usage data if user didn't request it
							const responseChunk = filterUsageFromChunk(
								typedChunk,
								userWantsUsage,
							);

							// Convert chunk to SSE format and enqueue
							const sseData = `data: ${JSON.stringify(responseChunk)}\n\n`;
							controller.enqueue(encoder.encode(sseData));
						}

						// Send [DONE] message
						controller.enqueue(encoder.encode("data: [DONE]\n\n"));
						controller.close();
						clearTimeout(timeoutId);
					} catch (error) {
						console.error("Streaming error:", error);
						clearTimeout(timeoutId); // Clear timeout on error

						const isAborted = abortController.signal.aborted;
						const errorMessage = isAborted ? "Stream timeout" : "Stream failed";
						const errorData = `data: ${JSON.stringify({ error: errorMessage })}\n\n`;
						controller.enqueue(encoder.encode(errorData));
						controller.close();

						// Record error usage
						queueMicrotask(async () => {
							try {
								await api.usage.recordApiUsage({
									apiKey,
									provider: null,
									model: null,
									usage: {
										promptTokens: 0,
										completionTokens: 0,
										totalTokens: 0,
									},
									duration: Date.now() - streamStartTime,
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
					Connection: "keep-alive",
					"Cache-Control": "no-cache, no-transform",
					"Content-Type": "text/event-stream; charset=utf-8",
				},
			});
		}
		// Non-streaming request
		const nonStreamStartTime = Date.now();

		try {
			const bodyWithProviders = {
				...internalBody,
				provider_configs: providerConfigs,
			};

			const completion = (await openai.chat.completions.create(internalBody, {
				body: bodyWithProviders,
			})) as ChatCompletion;

			// Record usage in background
			if (completion.usage) {
				queueMicrotask(async () => {
					try {
						await api.usage.recordApiUsage({
							apiKey,
							provider: completion.provider ?? null,
							model: completion.model,
							usage: {
								promptTokens: completion.usage?.prompt_tokens ?? 0,
								completionTokens: completion.usage?.completion_tokens ?? 0,
								totalTokens: completion.usage?.total_tokens ?? 0,
							},
							duration: Date.now() - nonStreamStartTime,
							timestamp: new Date(),
							cacheTier: completion.cache_tier,
						});
					} catch (error) {
						console.error("Failed to record usage:", error);
					}
				});
			}

			return Response.json(completion);
		} catch (error) {
			// Record error for non-streaming requests
			queueMicrotask(async () => {
				try {
					await api.usage.recordApiUsage({
						apiKey,
						provider: null,
						model: null,
						usage: {
							promptTokens: 0,
							completionTokens: 0,
							totalTokens: 0,
						},
						duration: Date.now() - nonStreamStartTime,
						timestamp: new Date(),
						requestCount: 1,
						error: error instanceof Error ? error.message : String(error),
					});
				} catch (err) {
					console.error("Failed to record error:", err);
				}
			});

			throw error; // Re-throw to be handled by outer catch
		}
	} catch (error) {
		console.log("Error: ", error);
		return new Response(JSON.stringify({ error: "Internal server error" }), {
			status: 500,
			headers: { "Content-Type": "application/json" },
		});
	}
}
