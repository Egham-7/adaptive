import { api } from "@/trpc/server";
import { ChatbotSidebar } from "../_components/chat-platform/chat-sidebar";
import PaymentNotificationWrapper from "../_components/landing_page/payment-notification";

export default async function HomeLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	const initialConversations = await api.conversations.list();
	return (
		<PaymentNotificationWrapper>
			<div className="flex min-h-screen w-full bg-background">
				<ChatbotSidebar initialConversations={initialConversations} />
				<main className="w-full flex-1 overflow-hidden px-10 py-2 pt-10">
					{children}
				</main>
			</div>
		</PaymentNotificationWrapper>
	);
}
