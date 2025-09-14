import type { NextRequest } from "next/server";
import OpenAI from "openai";
import { decryptProviderApiKey } from "@/lib/auth-utils";
import { createBackendJWT } from "@/lib/jwt";
import {
	filterUsageFromChunk,
	filterUsageFromCompletion,
	userRequestedUsage,
	withUsageTracking,
} from "@/lib/usage-utils";
import { api } from "@/trpc/server";
import type {
	ChatCompletion,
	ChatCompletionChunk,
	ChatCompletionRequest,
} from "@/types/chat-completion";
import { chatCompletionRequestSchema } from "@/types/chat-completion";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
	try {
		const rawBody = await req.json();

		// Validate and parse the request body
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
		const providerConfigs: Record<
			string,
			{
				base_url?: string;
				auth_type?: string;
				auth_header_name?: string;
				api_key?: string;
				health_endpoint?: string;
				rate_limit_rpm?: number;
				timeout_ms?: number;
				retry_config?: Record<string, unknown>;
				headers?: Record<string, string>;
			}
		> = {};

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
						auth_type: provider.authType ?? undefined,
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

		// Create JWT token for backend authentication
		const jwtToken = await createBackendJWT(apiKey);

		const openai = new OpenAI({
			apiKey: jwtToken, // Use JWT token instead of API key
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
					let finalCompletion: ChatCompletionChunk | null = null; // Minimize scope

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
							// Store final completion data for usage tracking
							if (chunk.usage || chunk.choices[0]?.finish_reason) {
								finalCompletion = chunk as ChatCompletionChunk;
							}

							// Filter out usage data if user didn't request it
							const responseChunk = filterUsageFromChunk(
								chunk as ChatCompletionChunk,
								userWantsUsage,
							);

							// Convert chunk to SSE format and enqueue
							const sseData = `data: ${JSON.stringify(responseChunk)}\n\n`;
							controller.enqueue(encoder.encode(sseData));
						}

						// Send [DONE] message
						controller.enqueue(encoder.encode("data: [DONE]\n\n"));
						controller.close();

						// Clear timeout on successful completion
						clearTimeout(timeoutId);

						// Record usage after stream completes
						if (finalCompletion?.usage) {
							const completion = finalCompletion; // Capture for closure
							queueMicrotask(async () => {
								try {
									await api.usage.recordApiUsage({
										apiKey,
										provider: completion.provider ?? null,
										model: completion.model ?? null,
										usage: {
											promptTokens: completion.usage?.prompt_tokens ?? 0,
											completionTokens:
												completion.usage?.completion_tokens ?? 0,
											totalTokens: completion.usage?.total_tokens ?? 0,
										},
										duration: Date.now() - streamStartTime,
										timestamp: new Date(),
										cacheTier: completion.usage?.cache_tier,
									});
								} catch (error) {
									console.error("Failed to record streaming usage:", error);
								}
							});
						}
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
					"Content-Encoding": "none",
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

			const completion = (await openai.chat.completions.create(
				bodyWithProviders,
				{
					body: {
						...internalBody,
						provider_configs: providerConfigs,
					},
				},
			)) as ChatCompletion;

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

			// Filter out usage data from response if user didn't request it
			const responseCompletion = filterUsageFromCompletion(
				completion,
				userWantsUsage,
			);

			return Response.json(responseCompletion);
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
