"use client";

import { useChat } from "ai/react";

import { Chat } from "@/components/ui/chat";

export default function ChatPlatformPage() {
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    append,
    status,
    stop,
  } = useChat({
    api: "/api/chat",
  });

  const isLoading = status === "streaming" || status === "submitted";

  return (
    <Chat
      className="h-full "
      messages={messages}
      input={input}
      handleInputChange={handleInputChange}
      handleSubmit={handleSubmit}
      isGenerating={isLoading}
      stop={stop}
      append={append}
      suggestions={[
        "Generate a tasty vegan lasagna recipe for 3 people.",
        "Generate a list of 5 questions for a frontend job interview.",
        "Who won the 2022 FIFA World Cup?",
      ]}
    />
  );
}
