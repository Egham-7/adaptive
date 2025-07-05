import { ChatMessage, type ChatMessageProps } from "./chat-message";
import { TypingLoader } from "./loader";
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
    <div className="mx-auto max-w-3xl px-4 space-y-4 pb-2">
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
      {isTyping && (
        <div className="w-full">
          <div className="bg-muted rounded-lg px-3 py-2 mr-auto max-w-3xl">
            <TypingLoader size="sm" className="opacity-70" />
          </div>
        </div>
      )}
    </div>
  );
}
