import { useEffect, useRef } from "react";
import { DBMessage } from "@/services/messages/types";
import { MessageItem } from "./message-list/message-item";
import StreamingMessage from "./message-list/streaming-message";
import { Loader2, AlertCircle } from "lucide-react";
import { ChatFooter } from "@/components/chat/chat-footer";
import { useSendMessage } from "@/lib/hooks/conversations/use-send-message";
import { EmptyState } from "./empty-state";
import Header from "./header";
import { useSidebar } from "../ui/sidebar";
import { useUpdateMessage } from "@/lib/hooks/conversations/use-update-message";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { convertToApiMessages } from "@/services/messages";

interface ChatProps {
  conversationId: number;
  messages: DBMessage[];
}

export function Chat({ conversationId, messages }: ChatProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const apiMessages = convertToApiMessages(messages);

  console.log("Messages: ", messages);

  const {
    sendMessage,
    isLoading: isSendingMessage,
    isStreaming: sendMessageIsStreaming,
    streamingContent: sendMessageStreamingContent,
    abortStreaming: sendMessageAbortStreaming,
    error: sendMessageError,
  } = useSendMessage(conversationId, apiMessages);

  const {
    streamingContent: updatedStreamingContent,
    updateMessage,
    isStreaming: updateMessageIsStreaming,
    abortStreaming: updateMessageAbortStreaming,
    error: updateMessageError,
  } = useUpdateMessage();

  // Derived states
  const isStreaming = sendMessageIsStreaming || updateMessageIsStreaming;
  const abortStreaming = sendMessageIsStreaming
    ? sendMessageAbortStreaming
    : updateMessageAbortStreaming;
  const streamingContent =
    sendMessageStreamingContent || updatedStreamingContent;
  const error = sendMessageError || updateMessageError;
  const isLoading = isSendingMessage;
  const { open, openMobile } = useSidebar();

  // Show empty state only when there are no messages and no streaming content
  const showEmptyState =
    messages.length === 0 && !isStreaming && !streamingContent;

  // Scroll to bottom when messages or streaming content changes
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingContent]);

  return (
    <>
      {!open && !openMobile && <Header />}
      <ScrollArea className="flex-1 w-full py-6 px-4">
        <div className="space-y-6">
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
                  updateMessage={updateMessage.mutateAsync}
                  isUpdating={isStreaming}
                />
              ))}
              {isStreaming && streamingContent && (
                <StreamingMessage content={streamingContent} />
              )}
            </>
          )}
          {isLoading && !isStreaming && !streamingContent && (
            <LoadingIndicator />
          )}
          {error && <ErrorMessage error={error} />}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>
      <ChatFooter
        isLoading={isLoading}
        sendMessage={sendMessage}
        isStreaming={isStreaming}
        abortStreaming={abortStreaming}
        isError={Boolean(error)}
      />
    </>
  );
}

function LoadingIndicator() {
  return (
    <Card className="max-w-[90%] mx-auto">
      <CardContent className="flex items-center gap-3 p-4">
        <Loader2 className="w-5 h-5 animate-spin text-primary" />
        <p>Processing your request...</p>
      </CardContent>
    </Card>
  );
}

function ErrorMessage({ error }: { error: Error | null }) {
  if (!error) return null;

  return (
    <Alert variant="destructive" className="max-w-[90%] mx-auto">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>
        {error.message || "An error occurred while processing your request."}
      </AlertDescription>
    </Alert>
  );
}

export default Chat;
