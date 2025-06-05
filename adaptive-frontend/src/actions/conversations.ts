"use server";

import { revalidatePath } from "next/cache";
import prisma from "@/lib/db";

// --- Types for Conversation Actions ---
export type CreateConversationData = {
  title: string;
  pinned?: boolean;
};

export type UpdateConversationData = {
  title?: string;
  pinned?: boolean;
};

// For return types, you can define them more explicitly if needed
// e.g., import { Conversation, Message } from "@prisma/client";
// export type ConversationWithMessages = Conversation & { messages: Message[] };
// export type CreateConversationReturnType = { success: boolean; conversation?: Conversation; error?: string; };

// --- Conversation CRUD Actions ---

export async function createConversation(data: CreateConversationData) {
  try {
    const conversation = await prisma.conversation.create({
      data: {
        title: data.title,
        pinned: data.pinned ?? false,
      },
    });
    revalidatePath("/conversations"); // Adjust as needed
    revalidatePath(`/chat/${conversation.id}`); // Adjust to your chat page route
    return { success: true, conversation };
  } catch (error) {
    console.error("Error creating conversation:", error);
    return { success: false, error: "Failed to create conversation." };
  }
}

export async function getConversation(id: number) {
  try {
    const conversation = await prisma.conversation.findUnique({
      where: { id, deletedAt: null },
      include: {
        messages: {
          where: { deletedAt: null },
          orderBy: { createdAt: "asc" },
        },
      },
    });
    if (!conversation) {
      return { success: false, error: "Conversation not found." };
    }
    return { success: true, conversation };
  } catch (error) {
    console.error("Error fetching conversation:", error);
    return { success: false, error: "Failed to fetch conversation." };
  }
}

export type GetConversationsOptions = {
  pinned?: boolean;
  includeMessages?: boolean;
};

export async function getConversations(options?: GetConversationsOptions) {
  try {
    const conversations = await prisma.conversation.findMany({
      where: {
        deletedAt: null,
        ...(options?.pinned !== undefined && { pinned: options.pinned }),
      },
      include: {
        messages: options?.includeMessages
          ? {
              where: { deletedAt: null },
              orderBy: { createdAt: "asc" },
            }
          : false,
      },
      orderBy: {
        updatedAt: "desc",
      },
    });
    return { success: true, conversations };
  } catch (error) {
    console.error("Error fetching conversations:", error);
    return { success: false, error: "Failed to fetch conversations." };
  }
}

export async function updateConversation(
  id: number,
  data: UpdateConversationData,
) {
  try {
    const updatedConversation = await prisma.conversation.update({
      where: { id, deletedAt: null },
      data: {
        ...(data.title !== undefined && { title: data.title }),
        ...(data.pinned !== undefined && { pinned: data.pinned }),
        updatedAt: new Date(), // Explicitly update timestamp
      },
    });
    revalidatePath("/conversations");
    revalidatePath(`/chat/${id}`);
    return { success: true, conversation: updatedConversation };
  } catch (error) {
    console.error("Error updating conversation:", error);
    return { success: false, error: "Failed to update conversation." };
  }
}

export async function deleteConversation(id: number) {
  try {
    const conversation = await prisma.conversation.update({
      where: { id, deletedAt: null },
      data: {
        deletedAt: new Date(),
      },
    });
    await prisma.message.updateMany({
      where: {
        conversationId: id,
        deletedAt: null,
      },
      data: {
        deletedAt: new Date(),
      },
    });
    revalidatePath("/conversations");
    revalidatePath(`/chat/${id}`, "layout");
    return { success: true, conversation };
  } catch (error) {
    console.error("Error deleting conversation:", error);
    return { success: false, error: "Failed to delete conversation." };
  }
}

export async function pinConversation(id: number) {
  return updateConversation(id, { pinned: true });
}

export async function unpinConversation(id: number) {
  return updateConversation(id, { pinned: false });
}
