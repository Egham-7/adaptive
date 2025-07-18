import type { NextRequest } from "next/server";
import OpenAI from "openai";
import { api } from "@/trpc/server";
import type {
	ChatCompletion,
	ChatCompletionRequest,
} from "@/types/chat-completion";

const openai = new OpenAI({
	apiKey: process.env.ADAPTIVE_API_KEY,
	baseURL: `${process.env.ADAPTIVE_API_BASE_URL}/v1`,
});

export async function POST(req: NextRequest) {
	try {
		const body: ChatCompletionRequest = await req.json();

		// Extract API key from X-Stainless-API-Key header
		const apiKey = req.headers.get("x-stainless-api-key");

		if (!apiKey) {
			return new Response(
				JSON.stringify({
					error: "API key required. Provide it via X-Stainless-API-Key header",
				}),
				{
					status: 401,
					headers: { "Content-Type": "application/json" },
				},
			);
		}

		const { valid } = await api.api_keys.verify({
			apiKey,
		});

		if (!valid) {
			return new Response(JSON.stringify({ error: "Invalid API key" }), {
				status: 401,
				headers: { "Content-Type": "application/json" },
			});
		}
		// Support both streaming and non-streaming requests
		const shouldStream = body.stream === true;

		if (shouldStream) {
			const stream = openai.chat.completions.stream({
				...body,
				stream: true,
			});

			const startTime = Date.now();

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
		const nonStreamStartTime = Date.now();

		try {
			const completion = (await openai.chat.completions.create(
				body,
			)) as ChatCompletion;

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
		console.error("Chat completions error:", error);
		return new Response(JSON.stringify({ error: "Internal server error" }), {
			status: 500,
			headers: { "Content-Type": "application/json" },
		});
	}
}
