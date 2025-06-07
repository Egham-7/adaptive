import { api } from "@/trpc/server";
import type { ConversationCreateOutput } from "@/types";
import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default async function ChatPlatformPage() {
  let newConversation: ConversationCreateOutput;
  try {
    // Create a new conversation using tRPC server-side
    newConversation = await api.conversations.create({
      title: "New Chat",
    });
  } catch (error) {
    console.error("Failed to create conversation:", error);
    throw error; // or redirect to an error page
  }
  // Redirect to the newly created conversation
  redirect(`/chat-platform/chats/${newConversation.id}`);
}
