import { api } from "@/trpc/server";
import { redirect } from "next/navigation";

export default async function ChatPlatformPage() {
	try {
		// Create a new conversation using tRPC server-side
		const newConversation = await api.conversations.create({
			title: "New Chat",
		});

		// Redirect to the newly created conversation
		redirect(`/chat-platform/chats/${newConversation.id}`);
	} catch (error) {
		// Handle error appropriately - could redirect to error page or show message
		console.error("Failed to create conversation:", error);
		throw error; // or redirect to an error page
	}
}
