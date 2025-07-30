"use client";

import { type UIMessage, useChat } from "@ai-sdk/react";
import { convertFileListToFileUIParts } from "ai";

// Extract part types from UIMessage
type MessagePart = UIMessage["parts"][number];

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
		experimental_throttle: 50,
		onFinish({ message }) {
			console.log("Message finished:", message);
		},
	});

	const handleInputChange = useCallback(
		(event: React.ChangeEvent<HTMLTextAreaElement>) => {
			setInput(event.target.value);
		},
		[],
	);

	const handleSuggestionSubmit = useCallback(
		async (text: string) => {
			if (hasReachedLimit) {
				return;
			}

			if (!text.trim()) return;

			sendMessage({
				role: "user",
				parts: [{ type: "text", text }],
			});
		},
		[sendMessage, hasReachedLimit],
	);

	const handleSubmit = useCallback(
		async (
			event?: { preventDefault?: () => void },
			options?: { files?: FileList },
		) => {
			event?.preventDefault?.();

			if (hasReachedLimit) {
				return;
			}

			if (!input.trim()) return;

			const parts: MessagePart[] = [{ type: "text", text: input }];

			if (options?.files && options.files.length > 0) {
				const fileParts = await convertFileListToFileUIParts(options.files);
				parts.push(...fileParts);
			}

			sendMessage({
				role: "user",
				parts,
			});
			setInput("");
		},
		[sendMessage, hasReachedLimit, input],
	);

	const isLoading = status === "streaming" || status === "submitted";
	const isError = status === "error";

	const handleRegenerate = useCallback(() => {
		regenerate();
	}, [regenerate]);

	const handleSendMessage = useCallback(
		async (message: { text: string }) => {
			sendMessage({
				role: "user",
				parts: [{ type: "text", text: message.text }],
			});
		},
		[sendMessage],
	);

	return (
		<Chat
			className="flex-1"
			showWelcomeInterface={true}
			messages={messages}
			input={input}
			handleInputChange={handleInputChange}
			handleSubmit={handleSubmit}
			handleSuggestionSubmit={handleSuggestionSubmit}
			setMessages={setMessages}
			sendMessage={handleSendMessage}
			isGenerating={isLoading}
			stop={stop}
			suggestions={CHAT_SUGGESTIONS as string[]}
			isError={isError}
			error={error}
			onRetry={handleRegenerate}
			hasReachedLimit={hasReachedLimit}
			remainingMessages={remainingMessages}
			isUnlimited={isUnlimited}
			limitsLoading={limitsLoading}
			userId={conversation.userId}
		/>
	);
}
