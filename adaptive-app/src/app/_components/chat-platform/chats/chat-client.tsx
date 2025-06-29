"use client";

import { type UIMessage, useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useState } from "react";

import { Chat } from "@/components/ui/chat";
import { useMessageLimits } from "@/hooks/messages/use-message-limits";
import type { ConversationListItem, Message } from "@/types";

const CHAT_SUGGESTIONS = [
	"Generate a tasty vegan lasagna recipe for 3 people.",
	"Generate a list of 5 questions for a frontend job interview.",
	"Who won the 2022 FIFA World Cup?",
];

interface ChatClientProps {
	conversation: ConversationListItem;
	initialMessages: Message[];
}

export function ChatClient({ conversation, initialMessages }: ChatClientProps) {
	const {
		isUnlimited,
		remainingMessages,
		hasReachedLimit,
		isLoading: limitsLoading,
	} = useMessageLimits();

	const mappedMessages = initialMessages as unknown as UIMessage[];

	const [input, setInput] = useState("");
	const {
		messages,
		setMessages,
		status,
		stop,
		error,
		sendMessage,
		regenerate,
	} = useChat({
		id: conversation.id.toString(),
		messages: mappedMessages,
		transport: new DefaultChatTransport({
			api: "/api/chat",
			credentials: "include",
		}),
	});

	const handleInputChange = useCallback(
		(event: React.ChangeEvent<HTMLTextAreaElement>) => {
			setInput(event.target.value);
		},
		[],
	);

	const handleSubmit = useCallback(
		(
			event?: { preventDefault?: () => void },
			options?: { files?: FileList },
		) => {
			event?.preventDefault?.();

			if (hasReachedLimit) {
				alert(
					"You've reached your daily message limit. Please upgrade to continue.",
				);
				return;
			}

			if (!input.trim()) return;

			sendMessage({ text: input, files: options?.files });
			setInput("");
		},
		[sendMessage, hasReachedLimit, input],
	);

	const isLoading = status === "streaming" || status === "submitted";
	const isError = status === "error";

	return (
		<div className="flex h-full flex-col">
			<Chat
				className="flex-1"
				showWelcomeInterface={true}
				messages={messages}
				input={input}
				handleInputChange={handleInputChange}
				handleSubmit={handleSubmit}
				setMessages={setMessages}
				isGenerating={isLoading}
				stop={stop}
				sendMessage={(message) => sendMessage({ text: message.text })}
				suggestions={CHAT_SUGGESTIONS as string[]}
				isError={isError}
				error={error}
				onRetry={regenerate}
				hasReachedLimit={hasReachedLimit}
				remainingMessages={remainingMessages}
				isUnlimited={isUnlimited}
				limitsLoading={limitsLoading}
				userId={conversation.userId}
			/>
		</div>
	);
}
