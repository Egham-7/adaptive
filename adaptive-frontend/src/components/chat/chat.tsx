import { useEffect, useRef } from "react";
import { DBMessage } from "@/services/messages/types";
import { MessageItem } from "./message-list/message-item";
import StreamingMessage from "./message-list/streaming-message";
import { Loader2 } from "lucide-react";
import { ChatFooter } from "@/components/chat/chat-footer";
import { Message } from "@/services/llms/types";
import { useSendMessage } from "@/lib/hooks/conversations/use-send-message";
import { EmptyState } from "./empty-state";

interface ChatProps {
  conversationId: number;
  messages: DBMessage[];
  apiMessages: Message[];
}

export function Chat({ conversationId, messages, apiMessages }: ChatProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Use the hook directly in this component
  const {
    sendMessage,
    isLoading: isSendingMessage,
    isStreaming,
    streamingContent,
    abortStreaming,
  } = useSendMessage(conversationId, apiMessages);

  // Combined loading state
  const isLoading = isSendingMessage;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Show empty state only when there are no messages and no streaming content
  const showEmptyState =
    messages.length === 0 && !isStreaming && !streamingContent;

  return (
    <>
      <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
        {showEmptyState ? (
          <EmptyState onSendMessage={sendMessage} isLoading={isLoading} />
        ) : (
          <>
            {messages.map((msg, index) => (
              <MessageItem
                key={msg.id}
                message={msg}
                conversationId={conversationId}
                index={index}
                messages={messages}
              />
            ))}
            {(isStreaming || streamingContent) && (
              <StreamingMessage content={streamingContent || ""} />
            )}
          </>
        )}
        {isLoading && !isStreaming && !streamingContent && <LoadingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      <ChatFooter
        isLoading={isLoading}
        sendMessage={sendMessage}
        isStreaming={isStreaming}
        abortStreaming={abortStreaming}
        isError={false}
      />
    </>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-muted p-4 max-w-[90%]">
      <Loader2 className="w-5 h-5 animate-spin text-primary" />
      <p>Processing your request...</p>
    </div>
  );
}

export default Chat;
