// src/app/api/chat/route.ts
import { createOpenAI } from "@ai-sdk/openai";
import {
  streamText,
  appendResponseMessages,
  type Message as SDKMessage,
  appendClientMessage,
} from "ai";
import { createCaller } from "@/server/api/root";
import { createTRPCContext } from "@/server/api/trpc";
import { TRPCError } from "@trpc/server";
import { messageRoleSchema } from "@/lib/chat/schema";
import z from "zod";

type MessageRole = z.infer<typeof messageRoleSchema>;

export async function POST(req: Request) {
  try {
    const { message, id: conversationId } = await req.json();

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

    const previousMesssages = await caller.messages.listByConversation({
      conversationId: numericConversationId,
    });
    const transformedMessages = previousMesssages.map((dbMessage) => ({
      id: dbMessage.id,
      role: dbMessage.role as MessageRole,
      content: dbMessage.content,
      createdAt: dbMessage.createdAt,
      reasoning: dbMessage.reasoning ?? undefined, // Convert null to undefined
      annotations: dbMessage.annotations
        ? JSON.parse(dbMessage.annotations)
        : undefined,
      parts: dbMessage.parts ? JSON.parse(dbMessage.parts) : undefined,
      experimental_attachments: dbMessage.experimentalAttachments
        ? JSON.parse(dbMessage.experimentalAttachments)
        : undefined,
    }));

    const currentMessagesFromClient = appendClientMessage({
      messages: transformedMessages,
      message,
    });

    const adaptive = createOpenAI({});

    const result = streamText({
      model: adaptive("gpt-4o-mini"), // Model will be auto selected based on the Adaptive API
      messages: currentMessagesFromClient,
      async onFinish({ response }) {
        console.log("Current Messages from Client:", currentMessagesFromClient);
        console.log("Response messages:", response.messages);
        const finalMessagesToPersistSDK: SDKMessage[] = appendResponseMessages({
          messages: currentMessagesFromClient as SDKMessage[],
          responseMessages: response.messages,
        });
        console.log("Final messages to persist:", finalMessagesToPersistSDK);
        const finalMessagesToPersist = finalMessagesToPersistSDK.map(
          (message) => {
            const {
              data,
              toolInvocations,
              experimental_attachments,
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
