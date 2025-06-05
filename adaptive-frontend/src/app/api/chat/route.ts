import { openai } from "@ai-sdk/openai";
import {
  streamText,
  CoreMessage,
  appendResponseMessages,
  Message as SDKMessage,
} from "ai";
import prisma from "@/lib/db";
import { batchUpsertMessages } from "@/actions/messages";

export const runtime = "edge";

export async function POST(req: Request) {
  try {
    const {
      messages: currentMessagesFromClient,
      id: conversationId,
    }: { messages: CoreMessage[]; id: string | number } = await req.json();

    const numericConversationId = Number(conversationId);
    if (isNaN(numericConversationId)) {
      return new Response("Invalid Conversation ID format", { status: 400 });
    }

    if (!numericConversationId) {
      return new Response("Conversation ID is required", { status: 400 });
    }

    const conversation = await prisma.conversation.findUnique({
      where: { id: numericConversationId, deletedAt: null },
      select: { id: true },
    });

    if (!conversation) {
      return new Response("Conversation not found", { status: 404 });
    }

    const result = streamText({
      model: openai("gpt-4o-mini"),
      messages: currentMessagesFromClient,
      async onFinish({ response }) {
        const finalMessagesToPersistSDK: SDKMessage[] = appendResponseMessages({
          messages: currentMessagesFromClient as SDKMessage[],
          responseMessages: response.messages,
        });

        const validMessages = finalMessagesToPersistSDK.filter((sdkMsg) => {
          if (sdkMsg.role === "data") {
            return false;
          }
          if (!sdkMsg.id) {
            console.warn(
              "SDK Message missing ID, cannot batch upsert:",
              sdkMsg,
            );
            return false;
          }
          return true;
        });

        if (validMessages.length > 0) {
          const batchResult = await batchUpsertMessages(
            numericConversationId,
            validMessages,
          );
          if (!batchResult.success) {
            console.error(
              `Batch upsert failed for conversation ${numericConversationId}: ${batchResult.error}`,
            );
          } else {
            console.log(
              `Successfully batch upserted ${batchResult.messages?.length || 0} messages for conversation ${numericConversationId}.`,
            );
          }
        } else {
          console.log(
            `No valid messages to batch upsert for conversation ${numericConversationId}.`,
          );
        }
      },
    });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Error in chat API:", error);
    if (error instanceof Error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(
      JSON.stringify({ error: "An unknown error occurred" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
