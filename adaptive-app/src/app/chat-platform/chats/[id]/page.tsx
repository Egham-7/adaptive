import { ChatClient } from "@/app/_components/chat-platform/chats/chat-client";
import { api } from "@/trpc/server";
import type { MessageListInput } from "@/types";
import { notFound } from "next/navigation";

interface ConversationPageProps {
	params: Promise<{
		id: string;
	}>;
}

export default async function ConversationPage({
	params,
}: ConversationPageProps) {
	const { id } = await params;
	const conversationIdNumber = Number(id);

	if (Number.isNaN(conversationIdNumber)) {
		notFound();
	}

	const conversation = await api.conversations.getById({
		id: conversationIdNumber,
	});
	const messages = await api.messages.listByConversation({
		conversationId: conversationIdNumber,
	});

	return <ChatClient conversation={conversation} initialMessages={messages} />;
}
