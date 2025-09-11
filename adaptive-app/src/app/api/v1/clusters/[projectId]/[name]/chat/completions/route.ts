import type { NextRequest } from "next/server";
import OpenAI from "openai";
import { decryptProviderApiKey } from "@/lib/auth-utils";
import { withCache } from "@/lib/cache-utils";
import { createBackendJWT } from "@/lib/jwt";
import { api } from "@/trpc/server";
import type {
	ChatCompletion,
	ChatCompletionChunk,
	ChatCompletionRequest,
} from "@/types/chat-completion";

// POST /api/v1/clusters/{projectId}/{name}/chat/completions - OpenAI-compatible chat completions with cluster routing
export async function POST(
	request: NextRequest,
	{ params }: { params: Promise<{ projectId: string; name: string }> },
) {
	try {
		const { projectId, name } = await params;
		const body: ChatCompletionRequest = await request.json();

		// Extract API key from OpenAI-compatible headers
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

		// Verify API key has access to this project
		if (verificationResult.projectId !== projectId) {
			return new Response(
				JSON.stringify({
					error: "API key does not have access to this project",
				}),
				{ status: 403, headers: { "Content-Type": "application/json" } },
			);
		}

		// Get cluster configuration with caching
		const cluster = await withCache(
			`cluster:${projectId}:${name}`,
			() => api.llmClusters.getByName({ projectId, name, apiKey }),
			300, // 5 minutes cache
		);

		if (!cluster) {
			return new Response(JSON.stringify({ error: "Cluster not found" }), {
				status: 404,
				headers: { "Content-Type": "application/json" },
			});
		}

		// Fetch provider configurations from database
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

		try {
			const configs = await withCache(
				`provider-configs:${projectId}`,
				() => api.providerConfigs.getAll({ projectId, apiKey }),
				60, // 1 minute cache (shorter for user configs)
			);

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
		}

		// Get full model details from provider models with caching
		const modelDetails = await withCache(
			`model-details:${cluster.id}`,
			async () => {
				// Process each cluster provider and get their model details
				const modelDetailsArray: Array<{
					provider: string;
					model_name: string;
					cost_per_1m_input_tokens: number;
					cost_per_1m_output_tokens: number;
					max_context_tokens: number;
					max_output_tokens?: number;
					supports_function_calling: boolean;
					languages_supported: string[];
					model_size_params?: string;
					latency_tier?: string;
					task_type?: string;
					complexity?: string;
				}> = [];

				// Fetch all provider models in parallel using Promise.allSettled
				const providerModelPromises = cluster.providers.map(
					async (clusterProvider) => {
						try {
							// Get models for this provider config
							const models = await api.providerModels.getForConfig({
								projectId,
								providerId: clusterProvider.providerId,
								configId: clusterProvider.configId ?? undefined,
								apiKey,
							});

							return { clusterProvider, models, error: null };
						} catch (error) {
							console.warn(
								`Failed to get models for provider ${clusterProvider.provider.name}:`,
								error,
							);
							return { clusterProvider, models: [], error };
						}
					},
				);

				// Wait for all provider model fetches to complete
				const providerResults = await Promise.allSettled(providerModelPromises);

				// Process results and build model details array
				for (const result of providerResults) {
					if (result.status === "fulfilled") {
						const { clusterProvider, models } = result.value;

						for (const model of models) {
							if (!model.capabilities) {
								console.warn(
									`Model ${model.name} from ${clusterProvider.provider.name} missing capabilities`,
								);
								continue;
							}

							modelDetailsArray.push({
								provider: clusterProvider.provider.name,
								model_name: model.name,
								cost_per_1m_input_tokens: Number(model.inputTokenCost),
								cost_per_1m_output_tokens: Number(model.outputTokenCost),
								max_context_tokens: model.capabilities.maxContextTokens ?? 4096,
								max_output_tokens:
									model.capabilities.maxOutputTokens ?? undefined,
								supports_function_calling:
									model.capabilities.supportsFunctionCalling,
								languages_supported:
									model.capabilities.languagesSupported ?? [],
								model_size_params:
									model.capabilities.modelSizeParams ?? undefined,
								latency_tier: model.capabilities.latencyTier ?? undefined,
								task_type: model.capabilities.taskType ?? undefined,
								complexity: model.capabilities.complexity ?? undefined,
							});
						}
					} else {
						console.warn(
							"Promise.allSettled rejected unexpectedly:",
							result.reason,
						);
					}
				}

				return modelDetailsArray;
			},
			300, // 5 minutes cache for model details
		);

		// Pre-flight credit check
		const messages = body.messages || [];
		const estimatedInputTokens = messages.reduce((acc, msg) => {
			return (
				acc + (typeof msg.content === "string" ? msg.content.length / 4 : 0)
			);
		}, 0);
		const estimatedOutputTokens = body.max_completion_tokens || 1000;

		try {
			await api.usage.checkCreditsBeforeUsage({
				apiKey,
				estimatedInputTokens,
				estimatedOutputTokens,
			});
		} catch (error: unknown) {
			const statusCode =
				(error as { code?: string }).code === "PAYMENT_REQUIRED" ? 402 : 400;
			return new Response(
				JSON.stringify({
					error:
						(error as { message?: string }).message || "Credit check failed",
				}),
				{
					status: statusCode,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		const shouldStream = body.stream === true;
		const baseURL = `${process.env.ADAPTIVE_API_BASE_URL}/v1`;
		const jwtToken = await createBackendJWT(apiKey);

		const openai = new OpenAI({
			apiKey: jwtToken,
			baseURL,
		});

		// Build enhanced request with cluster config and real model data
		const enhancedRequest = {
			...body,
			user: name,
			model_router: {
				models: modelDetails,
				cost_bias: cluster.costBias,
				complexity_threshold: cluster.complexityThreshold,
				token_threshold: cluster.tokenThreshold,
			},
			semantic_cache: {
				enabled: cluster.enableSemanticCache,
				semantic_threshold: cluster.semanticThreshold,
			},
			prompt_cache: {
				enabled: cluster.enablePromptCache,
				ttl: cluster.promptCacheTTL,
			},
			fallback: {
				enabled: cluster.fallbackEnabled,
				mode: cluster.fallbackMode,
			},
		};

		if (shouldStream) {
			const streamStartTime = Date.now();
			const _encoder = new TextEncoder();
			const abortController = new AbortController();
			const timeoutMs = 300000; // 5 minutes timeout

			const _timeoutId = setTimeout(() => {
				abortController.abort();
			}, timeoutMs);

			// Create custom ReadableStream that intercepts OpenAI SDK chunks
			const customReadable = new ReadableStream({
				async start(controller) {
					let finalChunk: ChatCompletionChunk | null = null;

					try {
						const stream = await openai.chat.completions.create(
							{
								...enhancedRequest,
								stream: true,
							},
							{
								body: {
									...enhancedRequest,
									stream: true,
									provider_configs: providerConfigs,
								},
							},
						);

						for await (const chunk of stream) {
							// Store final completion data for usage tracking
							if (chunk.usage || chunk.choices[0]?.finish_reason) {
								finalChunk = chunk as ChatCompletionChunk;
							}

							// Convert chunk to SSE format and enqueue
							const sseData = `data: ${JSON.stringify(chunk)}\n\n`;
							controller.enqueue(new TextEncoder().encode(sseData));
						}

						// Send [DONE] message
						controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
						controller.close();

						// Record usage after stream completes
						if (finalChunk?.usage) {
							queueMicrotask(async () => {
								try {
									await api.usage.recordApiUsage({
										apiKey,
										provider: finalChunk?.provider ?? null,
										model: finalChunk?.model ?? null,
										usage: {
											promptTokens: finalChunk?.usage?.prompt_tokens ?? 0,
											completionTokens:
												finalChunk?.usage?.completion_tokens ?? 0,
											totalTokens: finalChunk?.usage?.total_tokens ?? 0,
										},
										duration: Date.now() - streamStartTime,
										timestamp: new Date(),
										clusterId: cluster.id,
									});
								} catch (error) {
									console.error("Failed to record streaming usage:", error);
								}
							});
						}
					} catch (error) {
						console.error("Streaming error:", error);
						const errorData = `data: ${JSON.stringify({ error: "Stream failed" })}\n\n`;
						controller.enqueue(new TextEncoder().encode(errorData));
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
									clusterId: cluster.id,
								});
							} catch (err) {
								console.error("Failed to record error:", err);
							}
						});
					}
				},
			});

			return new Response(customReadable, {
				headers: {
					"Content-Type": "text/event-stream; charset=utf-8",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
					"Access-Control-Allow-Origin": "*",
				},
			});
		}

		// Non-streaming request
		const nonStreamStartTime = Date.now();

		try {
			const completion = (await openai.chat.completions.create(
				enhancedRequest,
				{
					body: {
						...enhancedRequest,
						provider_configs: providerConfigs,
					},
				},
			)) as ChatCompletion;

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
							clusterId: cluster.id,
						});
					} catch (error) {
						console.error("Failed to record usage:", error);
					}
				});
			}

			return Response.json(completion);
		} catch (error) {
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
						clusterId: cluster.id,
					});
				} catch (err) {
					console.error("Failed to record error:", err);
				}
			});

			throw error;
		}
	} catch (_error) {
		return new Response(JSON.stringify({ error: "Internal server error" }), {
			status: 500,
			headers: { "Content-Type": "application/json" },
		});
	}
}
