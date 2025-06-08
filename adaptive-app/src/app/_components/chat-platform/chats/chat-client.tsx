"use client";

import { Chat } from "@/components/ui/chat";
import type { ConversationListItem, Message } from "@/types";
import { useChat } from "ai/react";
import { useMemo } from "react";
import type { Message as SDKMessage } from "ai/react";

interface ChatClientProps {
  conversation: ConversationListItem;
  initialMessages: Message[];
}

export function ChatClient({ conversation, initialMessages }: ChatClientProps) {
  const mappedMessages = useMemo(
    () =>
      initialMessages.map((msg: Message) => ({
        id: msg.id,
        role: msg.role as "user" | "assistant" | "system",
        content: msg.content,
        createdAt: msg.createdAt,
        reasoning: msg.reasoning ?? undefined,
        annotations: msg.annotations ? msg.annotations : undefined,
        parts: msg.parts ? msg.parts : undefined,
        experimentalAttachments: msg.experimentalAttachments
          ? msg.experimentalAttachments
          : undefined,
      })) as unknown as SDKMessage[],
    [initialMessages],
  );

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    setMessages,
    append,
    status,
    stop,
    error,
    reload,
  } = useChat({
    api: "/api/chat",
    id: conversation.id.toString(),
    initialMessages: mappedMessages,
    experimental_prepareRequestBody({ messages, id }) {
      return { message: messages[messages.length - 1], id };
    },
  });

  const isLoading = status === "streaming" || status === "submitted";
  const isError = status === "error";

  return (
    <Chat
      className="h-full"
      messages={messages}
      input={input}
      handleInputChange={handleInputChange}
      handleSubmit={handleSubmit}
      setMessages={setMessages}
      isGenerating={isLoading}
      stop={stop}
      append={append}
      suggestions={[
        "Generate a tasty vegan lasagna recipe for 3 people.",
        "Generate a list of 5 questions for a frontend job interview.",
        "Who won the 2022 FIFA World Cup?",
      ]}
      isError={isError}
      error={error}
      onRetry={reload}
    />
  );
}
