// src/app/api/chat/route.ts
import { openai } from "@ai-sdk/openai";
import {
  streamText,
  type CoreMessage,
  appendResponseMessages,
  type Message as SDKMessage,
} from "ai";
import { createCaller } from "@/server/api/root";
import { createTRPCContext } from "@/server/api/trpc";
import { TRPCError } from "@trpc/server";
import { messageRoleSchema } from "@/lib/chat/schema";
import z from "zod";

type MessageRole = z.infer<typeof messageRoleSchema>;

export async function POST(req: Request) {
  try {
    const {
      messages: currentMessagesFromClient,
      id: conversationId,
    }: { messages: CoreMessage[]; id: string | number } = await req.json();

    const numericConversationId = Number(conversationId);

    if (isNaN(numericConversationId) || numericConversationId <= 0) {
      return new Response("Invalid Conversation ID", { status: 400 });
    }

    const trpcContext = await createTRPCContext({ headers: req.headers });
    if (!trpcContext.clerkAuth.userId) {
      return new Response("Unauthorized", { status: 401 });
    }

    const caller = createCaller(trpcContext);

    try {
      await caller.conversations.getById({ id: numericConversationId });
    } catch (error) {
      if (error instanceof TRPCError && error.code === "NOT_FOUND") {
        return new Response("Conversation not found or access denied", {
          status: 404,
        });
      }
      console.error("Error validating conversation via tRPC:", error);
      return new Response("Error validating conversation", { status: 500 });
    }

    const result = streamText({
      model: openai("gpt-4o-mini"),
      messages: currentMessagesFromClient,
      async onFinish({ response }) {
        const finalMessagesToPersistSDK: SDKMessage[] = appendResponseMessages({
          messages: currentMessagesFromClient as SDKMessage[],
          responseMessages: response.messages,
        });

        const finalMessagesToPersist = finalMessagesToPersistSDK.map(
          (message) => ({
            ...message,
            role: message.role as MessageRole,
            conversationId: numericConversationId,
            data: JSON.stringify(message.data) || null,
            annotations: JSON.stringify(message.annotations) || null,
            toolInvocations: JSON.stringify(message.toolInvocations) || null,
            parts: JSON.stringify(message.parts) || null,
          }),
        );

        await caller.messages.batchUpsert({
          conversationId: numericConversationId,
          messages: finalMessagesToPersist,
        });
      },
    });

    return result.toDataStreamResponse();
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
