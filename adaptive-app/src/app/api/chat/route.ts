import type { messageRoleSchema } from "@/lib/chat/schema";
import { api } from "@/trpc/server";
import type { Message as DBMessage } from "@/types";
import { createOpenAI } from "@ai-sdk/openai";
import { TRPCError } from "@trpc/server";
import {
	type Message as SDKMessage,
	appendClientMessage,
	appendResponseMessages,
	streamText,
} from "ai";
import OpenAI from "openai";
import type z from "zod";

async function logResponseChunks(response: Response) {
	if (!response.body) {
		console.log("No response body");
		return;
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) {
				console.log("Stream ended");
				break;
			}

			const chunk = decoder.decode(value, { stream: true });
			console.log("Chunk received:", chunk);
		}
	} catch (error) {
		console.error("Error reading stream:", error);
	} finally {
		reader.releaseLock();
	}
}

async function testOpenAIDirectly(messages: SDKMessage[]) {
	console.log("Testing OpenAI SDK directly...");

	const openai = new OpenAI({});

	try {
		// Filter out "data" role messages and map to OpenAI compatible format
		const openAIMessages = messages
			.filter((msg) => msg.role !== "data") // OpenAI doesn't support "data" role
			.map((msg) => ({
				role: msg.role as "system" | "user" | "assistant", // Explicitly type for OpenAI
				content: msg.content,
			}));

		const stream = await openai.chat.completions.create({
			model: "gpt-3.5-turbo", // This will be auto-selected by your API
			messages: openAIMessages,
			stream: true,
		});

		console.log("OpenAI SDK stream created successfully");

		for await (const chunk of stream) {
			console.log("OpenAI SDK chunk:", JSON.stringify(chunk, null, 2));
			if (chunk.choices[0]?.delta?.content) {
				console.log("Content delta:", chunk.choices[0].delta.content);
			}
		}
	} catch (error) {
		console.error("OpenAI SDK error:", error);
	}
}

type MessageRole = z.infer<typeof messageRoleSchema>;

export async function POST(req: Request) {
	try {
		const { message, id: conversationId } = await req.json();

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

		const previousMessages = (await api.messages.listByConversation({
			conversationId: numericConversationId,
		})) as DBMessage[];

		const transformedMessages: SDKMessage[] = previousMessages.map(
			(dbMessage: DBMessage) => ({
				id: dbMessage.id,
				role: dbMessage.role as MessageRole,
				content: dbMessage.content,
				createdAt: dbMessage.createdAt,
				reasoning: dbMessage.reasoning ?? undefined,
				annotations: dbMessage.annotations
					? JSON.parse(dbMessage.annotations)
					: undefined,
				parts: dbMessage.parts ? JSON.parse(dbMessage.parts) : undefined,
				experimental_attachments: dbMessage.experimentalAttachments
					? JSON.parse(dbMessage.experimentalAttachments)
					: undefined,
			}),
		);

		const currentMessagesFromClient = appendClientMessage({
			messages: transformedMessages,
			message,
		});

		await testOpenAIDirectly(currentMessagesFromClient);

		const adaptive = createOpenAI({
			baseURL: `${process.env.ADAPTIVE_API_BASE_URL}/api`,
		});

		const result = streamText({
			model: adaptive(""),
			messages: currentMessagesFromClient,
			async onFinish({ response }) {
				const finalMessagesToPersistSDK: SDKMessage[] = appendResponseMessages({
					messages: currentMessagesFromClient as SDKMessage[],
					responseMessages: response.messages,
				});
				const finalMessagesToPersist = finalMessagesToPersistSDK.map(
					(message: SDKMessage) => {
						const {
							experimental_attachments,
							// Remove deprecated properties
							...messageWithoutUnwantedProps
						} = message;
						return {
							...messageWithoutUnwantedProps,
							role: message.role as MessageRole,
							conversationId: numericConversationId,
							annotations: JSON.stringify(message.annotations) || null,
							parts: JSON.stringify(message.parts) || null,
							experimentalAttachments: experimental_attachments
								? JSON.stringify(experimental_attachments)
								: null,
						};
					},
				);

				await api.messages.batchUpsert({
					conversationId: numericConversationId,
					messages: finalMessagesToPersist,
				});
			},
		});

		console.log("Result: ", result);
		const data = result.toDataStreamResponse({
			sendUsage: true,
		});
		const dataText = result.toTextStreamResponse();

		console.log("Data: ", data);

		const response = dataText.clone();
		logResponseChunks(response);

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
