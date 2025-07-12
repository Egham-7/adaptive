import { ChatMessage, type ChatMessageProps } from "./chat-message";
import { TypingLoader } from "./loader";
import type { UIMessage } from "@ai-sdk/react";

type AdditionalMessageOptions = Omit<
  ChatMessageProps,
  keyof UIMessage
>;

interface MessageListProps {
  messages: UIMessage[];
  showTimeStamps?: boolean;
  messageOptions:
    | AdditionalMessageOptions
    | ((message: UIMessage) => AdditionalMessageOptions);
  isTyping?: boolean;
}

export function MessageList({
  messages,
  showTimeStamps = true,
  messageOptions,
  isTyping = false,
}: MessageListProps) {
  return (
    <div className="mx-auto max-w-3xl px-4 space-y-6 py-4">
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
      {isTyping && <TypingLoader size="sm" className="opacity-70" />}
    </div>
  );
}
