import type { NextRequest } from "next/server";
import OpenAI from "openai";
import { api } from "@/trpc/server";
import type {
	ChatCompletion,
	ChatCompletionRequest,
} from "@/types/chat-completion";

export async function POST(req: NextRequest) {
	try {
		console.log("=== Chat Completions API Request ===");
		console.log("Method:", req.method);
		console.log("URL:", req.url);

		const body: ChatCompletionRequest = await req.json();
		console.log("Request body model:", body.model);
		console.log("Request body stream:", body.stream);
		console.log("Request body messages count:", body.messages?.length);

		// Extract API key from OpenAI-compatible headers
		const authHeader = req.headers.get("authorization");
		console.log("Authorization header present:", !!authHeader);
		console.log(
			"Authorization header prefix:",
			`${authHeader?.substring(0, 10)}...`,
		);

		const bearerToken = authHeader?.startsWith("Bearer ")
			? authHeader.slice(7).replace(/\s+/g, "") || null
			: null;
		console.log("Bearer token extracted:", !!bearerToken);
		console.log("Bearer token length:", bearerToken?.length);

		const apiKey =
			bearerToken ||
			req.headers.get("x-api-key") ||
			req.headers.get("api-key") ||
			req.headers.get("x-stainless-api-key");

		console.log("Final API key found:", !!apiKey);
		console.log(
			"API key source:",
			bearerToken
				? "Bearer"
				: req.headers.get("x-api-key")
					? "X-API-Key"
					: req.headers.get("api-key")
						? "api-key"
						: req.headers.get("x-stainless-api-key")
							? "X-Stainless-API-Key"
							: "none",
		);

		if (!apiKey) {
			console.log("❌ No API key found, returning 401");
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

		console.log("🔍 Verifying API key...", `${apiKey.substring(0, 8)}...`);
		const { valid } = await api.api_keys.verify({
			apiKey,
		});
		console.log("API key verification result:", valid);

		if (!valid) {
			console.log("❌ Invalid API key, returning 401");
			return new Response(JSON.stringify({ error: "Invalid API key" }), {
				status: 401,
				headers: { "Content-Type": "application/json" },
			});
		}

		console.log("✅ API key valid, proceeding with request");
		// Support both streaming and non-streaming requests
		const shouldStream = body.stream === true;
		console.log("Stream mode:", shouldStream);

		const baseURL = `${process.env.ADAPTIVE_API_BASE_URL}/v1`;
		console.log("OpenAI client base URL:", baseURL);

		const openai = new OpenAI({
			apiKey,
			baseURL,
		});

		const openai = new OpenAI({
			apiKey,
			baseURL: `${process.env.ADAPTIVE_API_BASE_URL}/v1`,
		});

		if (shouldStream) {
			console.log("🌊 Starting streaming request...");
			console.log("Streaming request payload:", {
				model: body.model,
				messages: body.messages?.map((m) => ({
					role: m.role,
					content:
						typeof m.content === "string"
							? `${m.content.substring(0, 100)}...`
							: "[object]",
				})),
				stream: true,
				...Object.fromEntries(
					Object.entries(body).filter(([key]) => !["messages"].includes(key)),
				),
			});

			const stream = openai.chat.completions.stream({
				...body,
				stream: true,
			});

			const startTime = Date.now();
			console.log("Stream created, start time:", startTime);

			// Record usage when stream completes
			stream.on("finalChatCompletion", (completion) => {
				const adaptiveCompletion = completion as ChatCompletion;
				if (adaptiveCompletion.usage) {
					// Use setImmediate for zero-blocking tRPC call
					setImmediate(async () => {
						try {
							await api.usage.recordApiUsage({
								apiKey,
								provider: adaptiveCompletion.provider,
								model: adaptiveCompletion.model,
								usage: {
									promptTokens: adaptiveCompletion.usage?.prompt_tokens ?? 0,
									completionTokens:
										adaptiveCompletion.usage?.completion_tokens ?? 0,
									totalTokens: adaptiveCompletion.usage?.total_tokens ?? 0,
								},
								duration: Date.now() - startTime,
								timestamp: new Date(),
							});
						} catch (error) {
							console.error("Failed to record usage:", error);
							// Silent failure - never affect client stream
						}
					});
				}
			});

			// Handle stream errors
			stream.on("error", (error) => {
				setImmediate(async () => {
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
							duration: Date.now() - startTime,
							timestamp: new Date(),
							requestCount: 1,
							error: error.message,
						});
					} catch (err) {
						console.error("Failed to record error:", err);
					}
				});
			});

			return new Response(stream.toReadableStream(), {
				headers: {
					"Content-Type": "text/plain; charset=utf-8",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
					"Access-Control-Allow-Origin": "*",
				},
			});
		}
		// Non-streaming request
		console.log("📝 Starting non-streaming request...");
		console.log("Non-streaming request payload:", {
			model: body.model,
			messages: body.messages?.map((m) => ({
				role: m.role,
				content:
					typeof m.content === "string"
						? `${m.content.substring(0, 100)}...`
						: "[object]",
			})),
			...Object.fromEntries(
				Object.entries(body).filter(([key]) => !["messages"].includes(key)),
			),
		});

		const nonStreamStartTime = Date.now();
		console.log("Non-stream start time:", nonStreamStartTime);

		try {
			console.log("🚀 Making OpenAI completion request...");
			const completion = (await openai.chat.completions.create(
				body,
			)) as ChatCompletion;
			console.log("✅ Completion received:", {
				id: completion.id,
				model: completion.model,
				provider: completion.provider,
				usage: completion.usage,
			});

			// Record usage in background
			if (completion.usage) {
				setImmediate(async () => {
					try {
						await api.usage.recordApiUsage({
							apiKey,
							provider: completion.provider,
							model: completion.model,
							usage: {
								promptTokens: completion.usage?.prompt_tokens ?? 0,
								completionTokens: completion.usage?.completion_tokens ?? 0,
								totalTokens: completion.usage?.total_tokens ?? 0,
							},
							duration: Date.now() - nonStreamStartTime,
							timestamp: new Date(),
						});
					} catch (error) {
						console.error("Failed to record usage:", error);
					}
				});
			}

			return Response.json(completion);
		} catch (error) {
			// Record error for non-streaming requests
			setImmediate(async () => {
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
		console.error("❌ Chat completions outer error:", error);
		console.error("Error type:", typeof error);
		console.error(
			"Error name:",
			error instanceof Error ? error.name : "Unknown",
		);
		console.error(
			"Error message:",
			error instanceof Error ? error.message : String(error),
		);
		if (error instanceof Error && error.stack) {
			console.error("Error stack:", error.stack);
		}
		return new Response(JSON.stringify({ error: "Internal server error" }), {
			status: 500,
			headers: { "Content-Type": "application/json" },
		});
	}
}
