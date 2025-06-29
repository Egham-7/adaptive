import { ChatMessage, type ChatMessageProps } from "./chat-message";
import { TypingIndicator } from "./typing-indicator";
import type { UIMessage } from "@ai-sdk/react";

type AdditionalMessageOptions = Omit<ChatMessageProps, keyof UIMessage>;

interface MessageListProps {
  messages: UIMessage[];
  showTimeStamps?: boolean;
  isTyping?: boolean;
  messageOptions:
    | AdditionalMessageOptions
    | ((message: UIMessage) => AdditionalMessageOptions);
}

export function MessageList({
  messages,
  showTimeStamps = true,
  isTyping = false,
  messageOptions,
}: MessageListProps) {
  return (
    <div className="space-y-4 overflow-visible">
      {messages.map((message) => {
        const additionalOptions =
          typeof messageOptions === "function"
            ? messageOptions(message)
            : messageOptions;

        return (
          <ChatMessage
            key={message.id}
            showTimeStamp={showTimeStamps}
            {...message}
            {...additionalOptions}
          />
        );
      })}
      {isTyping && <TypingIndicator />}
    </div>
  );
}
