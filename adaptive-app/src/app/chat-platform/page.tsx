import { api } from "@/trpc/server";
import { redirect } from "next/navigation";

export default async function ChatPlatformPage() {
	// Create a new conversation using tRPC server-side
	const newConversation = await api.conversations.create({
		title: "New Chat",
	});

	// Redirect to the newly created conversation
	redirect(`/chat-platform/chats/${newConversation.id}`);
}
