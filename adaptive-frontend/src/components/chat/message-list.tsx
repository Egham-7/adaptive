import { Loader2, MessageSquare } from "lucide-react";
import { useEffect, useRef } from "react";
import { DBMessage } from "@/services/messages/types";
import { MessageItem } from "./message-list/message-item";
import StreamingMessage from "./message-list/streaming-message";

interface MessageListProps {
  conversationId: number;
  messages: DBMessage[];
  isLoading: boolean;
  error: string | null;
  streamingContent: string;
  isStreaming: boolean;
}

function MessageList({
  conversationId,
  messages,
  isLoading,
  error,
  isStreaming = false,
  streamingContent = "",
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  return (
    <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
      {messages.length === 0 && !isStreaming ? (
        <EmptyState />
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

          {isStreaming && streamingContent && (
            <StreamingMessage content={streamingContent} />
          )}
        </>
      )}

      {isLoading && !isStreaming && <LoadingIndicator />}
      {error && <ErrorMessage error={error} />}
      <div ref={messagesEndRef} />
    </div>
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

function ErrorMessage({ error }: { error: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-destructive/10 p-4 max-w-[90%] text-destructive border border-destructive/20">
      <p>Error: {error}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center w-full text-center">
      <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
        <MessageSquare className="w-8 h-8 text-primary" />
      </div>
      <h2 className="mb-2 text-xl font-medium">Welcome to Adaptive</h2>
      <div className="max-w-md text-muted-foreground">
        Send a message to start a conversation. Adaptive will select the best AI
        model for your specific query.
      </div>
    </div>
  );
}

export default MessageList;
