import { hasReachedDailyLimit } from "@/lib/chat/message-limits";
import type { messageRoleSchema } from "@/lib/chat/schema";
import { isUserSubscribed } from "@/lib/stripe/subscription-utils";
import { db } from "@/server/db";
import { api } from "@/trpc/server";
import { createOpenAI } from "@ai-sdk/openai";
import { auth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import { convertToModelMessages, streamText, type UIMessage } from "ai";
import type { z } from "zod";

type MessageRole = z.infer<typeof messageRoleSchema>;

export async function POST(req: Request) {
  try {
    const { userId } = await auth();

    if (!userId) {
      return new Response("Unauthorized", { status: 401 });
    }

    const body = await req.json();
    console.log("Received body:", body);
    const { messages, id: conversationId } = body;

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
    const isSubscribed = await isUserSubscribed(db, userId);

    // If not subscribed, check daily limit before processing
    if (!isSubscribed) {
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
    })) as UIMessage[];

    // Convert UI messages to core messages for the AI model
    const coreMessages = convertToModelMessages([
      ...previousMessages,
      ...messages,
    ]);
    const adaptive = createOpenAI({
      baseURL: `${process.env.ADAPTIVE_API_BASE_URL}/v1`,
      name: "Adaptive AI",
    });

    const result = streamText({
      model: adaptive.chat(""),
      messages: coreMessages,
      async onFinish({ text }) {
        // Create the assistant response message
        const assistantMessage = {
          id: crypto.randomUUID(),
          role: "assistant" as MessageRole,
          content: text,
          conversationId: numericConversationId,
          parts: [{ type: "text" as const, text }],
          metadata: null,
          annotations: null,
        };

        const message = messages[messages.length - 1];

        // Also save the user message if it's new
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

        await api.messages.batchUpsert({
          conversationId: numericConversationId,
          messages: [userMessage, assistantMessage],
        });
      },
    });

    const data = result.toUIMessageStreamResponse();

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
