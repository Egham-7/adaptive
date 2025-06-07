import { api } from "@/trpc/server";
import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default async function ChatPlatformPage() {
  const newConversation = await api.conversations.create({
    title: "New Chat",
  });
  // Redirect to the newly created conversation
  redirect(`/chat-platform/chats/${newConversation.id}`);
}
