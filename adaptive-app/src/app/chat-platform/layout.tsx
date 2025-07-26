import { api } from "@/trpc/server";
import { ChatbotSidebar } from "../_components/chat-platform/chat-sidebar";

export default async function HomeLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	const initialConversations = await api.conversations.list();
	return (
		<div className="flex min-h-screen w-full bg-background">
			<ChatbotSidebar initialConversations={initialConversations} />
			<main className="w-full flex-1 overflow-hidden px-10 py-2 pt-10">
				{children}
			</main>
		</div>
	);
}
