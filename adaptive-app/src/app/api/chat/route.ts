import { createAdaptive } from "@adaptive-llm/adaptive-ai-provider";
import { auth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import {
	convertToModelMessages,
	generateText,
	stepCountIs,
	streamText,
	tool,
	type UIMessage,
} from "ai";
import { Exa } from "exa-js";
import type { z } from "zod";
import { z as zodSchema } from "zod";
import { hasReachedDailyLimit } from "@/lib/chat/message-limits";
import type { messageRoleSchema } from "@/lib/chat/schema";
import { db } from "@/server/db";
import { api } from "@/trpc/server";

type MessageRole = z.infer<typeof messageRoleSchema>;

// Validate required environment variables
if (!process.env.ADAPTIVE_API_KEY) {
	throw new Error(
		"ADAPTIVE_API_KEY environment variable is required but not defined",
	);
}

if (!process.env.ADAPTIVE_API_BASE_URL) {
	throw new Error(
		"ADAPTIVE_API_BASE_URL environment variable is required but not defined",
	);
}

const adaptive = createAdaptive({
	apiKey: process.env.ADAPTIVE_API_KEY,
	baseURL: `${process.env.ADAPTIVE_API_BASE_URL}/v1`,
});

// Web search function using Exa API
async function webSearch(query: string): Promise<
	Array<{
		title: string;
		url: string;
		snippet: string;
		publishedDate?: string;
	}>
> {
	try {
		const exaApiKey = process.env.EXA_API_KEY;
		if (!exaApiKey || exaApiKey.trim() === "") {
			throw new Error(
				"EXA_API_KEY environment variable is not defined or empty. Please configure your Exa API key.",
			);
		}

		const exa = new Exa(exaApiKey);

		const searchResponse = await exa.searchAndContents(query, {
			type: "neural",
			useAutoprompt: true,
			numResults: 5,
			text: true,
		});

		return searchResponse.results.map((result) => ({
			title: result.title || "Untitled",
			url: result.url,
			snippet: result.text || result.title || "No snippet available",
			publishedDate: result.publishedDate,
		}));
	} catch (error) {
		console.error("Web search error:", error);
		throw error;
	}
}

export async function POST(req: Request) {
	try {
		const { userId } = await auth();

		if (!userId) {
			return new Response("Unauthorized", { status: 401 });
		}

		const body = await req.json();
		const { messages, id: conversationId, searchEnabled = false } = body;

		const numericConversationId = Number(conversationId);

		if (Number.isNaN(numericConversationId) || numericConversationId <= 0) {
			return new Response("Invalid Conversation ID", { status: 400 });
		}

		try {
			await api.conversations.getById({ id: numericConversationId });
		} catch (error) {
			if (error instanceof TRPCError && error.code === "NOT_FOUND") {
				return new Response("Conversation not found or access denied", {
					status: 404,
				});
			}
			console.error("Error validating conversation via tRPC:", error);
			return new Response("Error validating conversation", { status: 500 });
		}

		// Check if user is subscribed
		const subscription = await db.subscription.findFirst({
			where: {
				userId: userId,
				status: "active",
			},
		});
		const isSubscribed = !!subscription;

		// If not subscribed, check daily limit before processing (skip in development)
		const isDevelopment = process.env.NODE_ENV === "development";
		if (!isSubscribed && !isDevelopment) {
			const hasReachedLimit = await hasReachedDailyLimit(db, userId);
			if (hasReachedLimit) {
				return new Response(
					JSON.stringify({
						error: "Daily message limit reached. Please upgrade to continue.",
					}),
					{
						status: 403,
						headers: { "Content-Type": "application/json" },
					},
				);
			}
		}

		const previousMessages = (await api.messages.listByConversation({
			conversationId: numericConversationId,
		})) as unknown as UIMessage[];

		// Convert UI messages to core messages for the AI model
		const coreMessages = convertToModelMessages([
			...previousMessages,
			...messages,
		]);

		// Save user message immediately before attempting AI response
		const message = messages[messages.length - 1];
		const userMessage = {
			id: message.id || crypto.randomUUID(),
			role: message.role as MessageRole,
			conversationId: numericConversationId,
			parts: message.parts || [
				{ type: "text" as const, text: message.content as string },
			],
			metadata: message.metadata ?? null,
			annotations: message.annotations ?? null,
		};

		await api.messages.create(userMessage);

		// Check if this is the first message in the conversation to generate a title
		const isFirstMessage = previousMessages.length === 0;
		const shouldGenerateTitle = isFirstMessage && message.content;

		const tools = searchEnabled
			? {
					webSearch: tool({
						description:
							"Search the web for current information, news, facts, or any topic that requires up-to-date information",
						inputSchema: zodSchema.object({
							query: zodSchema
								.string()
								.describe("The search query to look up on the web"),
						}),
						execute: async ({ query }) => {
							const searchResults = await webSearch(query);
							return {
								query,
								results: searchResults,
							};
						},
					}),
				}
			: undefined;

		console.log("Tools: ", tools);

		let provider: string | undefined;
		let modelId: string | undefined;

		const result = streamText({
			model: adaptive.chat(),
			messages: coreMessages,
			tools,
			async onFinish({ text, providerMetadata, response, usage }) {
				// Create the assistant response message
				const assistantMessage = {
					id: crypto.randomUUID(),
					role: "assistant" as MessageRole,
					content: text,
					conversationId: numericConversationId,
					parts: [{ type: "text" as const, text }],
					metadata: {
						providerMetadata,
						response,
						usage,
					},
					annotations: null,
				};

				await api.messages.create(assistantMessage);

				// Generate title for the first message
				if (shouldGenerateTitle) {
					try {
						const titleResult = await generateText({
							model: adaptive.chat(),
							messages: [
								{
									role: "system",
									content:
										"Generate a concise, descriptive title (2-6 words) for the following conversation based on the user's message. Return only the title, no quotes or extra text.",
								},
								{
									role: "user",
									content: message.content as string,
								},
							],
						});

						// Clean up the title and ensure it's not too long
						const cleanTitle = titleResult.text
							.trim()
							.replace(/^["']|["']$/g, "");
						const finalTitle =
							cleanTitle.length > 50
								? `${cleanTitle.slice(0, 50)}...`
								: cleanTitle;

						// Update the conversation with the generated title
						if (finalTitle) {
							await api.conversations.update({
								id: numericConversationId,
								title: finalTitle,
							});
						}
					} catch (error) {
						console.error("Error generating conversation title:", error);
						// Title generation failure shouldn't affect the chat response
					}
				}

				provider = providerMetadata?.adaptive?.provider as string | undefined;
				modelId = response.modelId || undefined;
			},
			stopWhen: stepCountIs(5),
		});

		const data = result.toUIMessageStreamResponse({
			sendReasoning: true,
			sendSources: true,
			messageMetadata: ({ part }) => {
				return {
					...part,
					provider,
					modelId,
				};
			},
		});

		return data;
	} catch (error) {
		console.error("Error in chat API:", error);
		const errorMessage =
			error instanceof Error ? error.message : "Unknown error";
		return new Response(JSON.stringify({ error: errorMessage }), {
			status: 500,
			headers: { "Content-Type": "application/json" },
		});
	}
}
