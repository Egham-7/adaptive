"use server";

import { revalidatePath } from "next/cache";
import prisma from "@/lib/db";
import { Message, Prisma } from "@prisma/client";

export type MessageRole = "system" | "user" | "assistant" | "data";

// --- Types for Message Actions ---
export type CreateMessageData = {
  conversationId: number;
  role: MessageRole;
  content: string;
  id?: string;
  createdAt?: Date | string;
  reasoning?: string | null;
  data?: string | null;
  annotations?: string | null;
  toolInvocations?: string | null;
  parts?: string | null;
  experimentalAttachments?: string | null;
};

export type UpdateMessageData = {
  content?: string;
  reasoning?: string | null;
  data?: string | null;
  annotations?: string | null;
  toolInvocations?: string | null;
  parts?: string | null;
  experimentalAttachments?: string | null;
};

export type BatchUpsertMessageData = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: Date | string;
  reasoning?: string | null;
  data?: string | null;
  annotations?: string | null;
  toolInvocations?: string | null;
  parts?: string | null;
  experimentalAttachments?: string | null;
};

// --- Message CRUD Actions ---

export async function createMessage(data: CreateMessageData) {
  try {
    const conversation = await prisma.conversation.findUnique({
      where: { id: data.conversationId, deletedAt: null },
      select: { id: true },
    });
    if (!conversation) {
      return {
        success: false,
        error: "Conversation not found or has been deleted.",
      };
    }
    const messageDataToCreate: Prisma.MessageCreateInput = {
      role: data.role,
      content: data.content,
      reasoning: data.reasoning,
      data: data.data,
      annotations: data.annotations,
      toolInvocations: data.toolInvocations,
      parts: data.parts,
      experimentalAttachments: data.experimentalAttachments,
      conversation: { connect: { id: data.conversationId } },
      ...(data.createdAt && { createdAt: new Date(data.createdAt) }),
    };
    const message = await prisma.message.create({ data: messageDataToCreate });
    await prisma.conversation.update({
      where: { id: data.conversationId },
      data: { updatedAt: new Date() },
    });
    revalidatePath(`/chat/${data.conversationId}`);
    return { success: true, message };
  } catch (error) {
    console.error("Error creating message:", error);
    return { success: false, error: "Failed to create message." };
  }
}

export async function getMessages(conversationId: number) {
  try {
    const conversationExists = await prisma.conversation.count({
      where: { id: conversationId, deletedAt: null },
    });
    if (conversationExists === 0) {
      return { success: false, error: "Conversation not found." };
    }
    const messages = await prisma.message.findMany({
      where: { conversationId: conversationId, deletedAt: null },
      orderBy: { createdAt: "asc" },
    });
    return { success: true, messages };
  } catch (error) {
    console.error("Error fetching messages:", error);
    return { success: false, error: "Failed to fetch messages." };
  }
}

export async function getMessage(id: string) {
  try {
    const message = await prisma.message.findUnique({
      where: { id, deletedAt: null },
    });
    if (!message) {
      return { success: false, error: "Message not found." };
    }
    return { success: true, message };
  } catch (error) {
    console.error("Error fetching message:", error);
    return { success: false, error: "Failed to fetch message." };
  }
}

export async function updateMessage(id: string, data: UpdateMessageData) {
  try {
    const messageToUpdate = await prisma.message.findUnique({
      where: { id, deletedAt: null },
      select: { conversationId: true },
    });
    if (!messageToUpdate) {
      return { success: false, error: "Message not found to update." };
    }
    const updatedMessage = await prisma.message.update({
      where: { id },
      data: {
        ...(data.content !== undefined && { content: data.content }),
        ...(data.reasoning !== undefined && { reasoning: data.reasoning }),
        ...(data.data !== undefined && { data: data.data }),
        ...(data.annotations !== undefined && {
          annotations: data.annotations,
        }),
        ...(data.toolInvocations !== undefined && {
          toolInvocations: data.toolInvocations,
        }),
        ...(data.parts !== undefined && { parts: data.parts }),
        ...(data.experimentalAttachments !== undefined && {
          experimentalAttachments: data.experimentalAttachments,
        }),
        updatedAt: new Date(),
      },
    });
    await prisma.conversation.update({
      where: { id: messageToUpdate.conversationId },
      data: { updatedAt: new Date() },
    });
    revalidatePath(`/chat/${messageToUpdate.conversationId}`);
    return { success: true, message: updatedMessage };
  } catch (error) {
    console.error("Error updating message:", error);
    return { success: false, error: "Failed to update message." };
  }
}

export async function deleteMessage(id: string) {
  try {
    const messageToDelete = await prisma.message.findUnique({
      where: { id, deletedAt: null },
      select: { conversationId: true },
    });
    if (!messageToDelete) {
      return { success: false, error: "Message not found to delete." };
    }
    const deletedMessage = await prisma.message.update({
      where: { id },
      data: { deletedAt: new Date() },
    });
    await prisma.conversation.update({
      where: { id: messageToDelete.conversationId },
      data: { updatedAt: new Date() },
    });
    revalidatePath(`/chat/${messageToDelete.conversationId}`);
    return { success: true, message: deletedMessage };
  } catch (error) {
    console.error("Error deleting message:", error);
    return { success: false, error: "Failed to delete message." };
  }
}

export async function batchUpsertMessages(
  conversationId: number,
  messagesData: BatchUpsertMessageData[],
) {
  if (!messagesData || messagesData.length === 0) {
    return { success: true, messages: [], error: null };
  }
  const upsertOperations: Prisma.PrismaPromise<Message>[] = [];
  const validationErrors: string[] = [];

  for (const msgData of messagesData) {
    if (!msgData.id) {
      validationErrors.push("A message is missing an ID for batch upsert.");
      continue;
    }
    const messageToUpsert = {
      role: msgData.role,
      content: msgData.content,
      reasoning: msgData.reasoning,
      data: msgData.data,
      annotations: msgData.annotations,
      toolInvocations: msgData.toolInvocations,
      parts: msgData.parts,
      experimentalAttachments: msgData.experimentalAttachments,
    };
    upsertOperations.push(
      prisma.message.upsert({
        where: { id: msgData.id },
        create: {
          id: msgData.id,
          conversationId: conversationId,
          createdAt: msgData.createdAt
            ? new Date(msgData.createdAt)
            : new Date(),
          ...messageToUpsert,
        },
        update: { ...messageToUpsert, updatedAt: new Date() },
      }),
    );
  }

  if (validationErrors.length > 0) {
    return {
      success: false,
      messages: null,
      error: `Validation errors: ${validationErrors.join("; ")}`,
    };
  }
  if (upsertOperations.length === 0) {
    return {
      success: true,
      messages: [],
      error: "No valid messages to process after validation.",
    };
  }

  try {
    const results = await prisma.$transaction(upsertOperations);
    await prisma.conversation.update({
      where: { id: conversationId },
      data: { updatedAt: new Date() },
    });
    revalidatePath(`/chat/${conversationId}`);
    return { success: true, messages: results, error: null };
  } catch (error) {
    console.error(
      `Error in batchUpsertMessages for conversation ${conversationId}:`,
      error,
    );
    return {
      success: false,
      messages: null,
      error: `Failed to batch upsert messages. Details: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}
