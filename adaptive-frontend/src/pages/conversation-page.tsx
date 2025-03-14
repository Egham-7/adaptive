import { useState } from "react";
import { useParams } from "@tanstack/react-router";
import { useConversationState } from "@/lib/hooks/conversations/use-conversation-state";
import { useSendMessage } from "@/lib/hooks/conversations/use-send-message";
import { useUpdateConversation } from "@/lib/hooks/conversations/use-update-conversation";
import { useDeleteConversationMessages } from "@/lib/hooks/conversations/use-delete-conversation-messages";
import { useConversationMessages } from "@/lib/hooks/conversations/use-conversation-message";
import { useConversation } from "@/lib/hooks/conversations/use-conversation";

// Components
import { ChatFooter } from "@/components/chat/chat-footer";
import { ChatHeader } from "@/components/chat/chat-header";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, RefreshCw, Cpu, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import MessageList from "@/components/chat/message-list";
import { cn } from "@/lib/utils";
import { convertToApiMessages } from "@/services/messages";

export default function ConversationPage() {
  const [showActions, setShowActions] = useState(true);

  const { conversationId } = useParams({
    from: "/_home/conversations/$conversationId",
  });

  const numericConversationId = Number(conversationId);

  // Fetch conversation data
  const {
    data: conversation,
    isLoading: isLoadingConversation,
    error: conversationError,
    refetch: refetchConversation,
  } = useConversation(numericConversationId);

  // Fetch conversation messages
  const {
    data: messages = [],
    isLoading: isLoadingMessages,
    error: messagesError,
    refetch: refetchMessages,
  } = useConversationMessages(numericConversationId);

  const chatCompletionMessages = convertToApiMessages(messages);

  // Manage conversation state
  const { title, setTitle: setStateTitle } = useConversationState({
    conversationId: numericConversationId,
    initialTitle: conversation?.title,
  });

  // Update conversation
  const {
    mutateAsync: updateConversation,
    isPending: isUpdatingConversation,
    error: updateError,
  } = useUpdateConversation();

  // Delete messages
  const { mutateAsync: deleteMessages } = useDeleteConversationMessages();

  // Send message
  const {
    sendMessage: originalSendMessage,
    isLoading: isSendingMessage,
    error: sendError,
    isStreaming,
    streamingContent,
    modelInfo,
    abortStreaming,
  } = useSendMessage(numericConversationId, chatCompletionMessages);

  // Action functions
  const setTitle = async (newTitle: string) => {
    setStateTitle(newTitle);
    await updateConversation({
      id: numericConversationId,
      title: newTitle,
    });
  };

  const sendMessage = async (content: string) => {
    const response = await originalSendMessage(content);
    return response;
  };

  const resetConversation = async () => {
    if (isStreaming) {
      abortStreaming();
    }
    await deleteMessages(numericConversationId);
  };

  // Combined loading state for footer and other UI elements
  const isLoading =
    isLoadingConversation ||
    isSendingMessage ||
    isUpdatingConversation ||
    isLoadingMessages;

  const initialLoading = isLoadingConversation || isLoadingMessages;

  // Toggle action buttons visibility
  const toggleActions = () => {
    setShowActions(!showActions);
  };

  // Get current model info from the last response
  const currentProvider = modelInfo?.provider;
  const currentModel = modelInfo?.model;

  // Retry function for errors
  const handleRetry = () => {
    refetchConversation();
    refetchMessages();
  };

  // Determine what error to show
  const getErrorContent = () => {
    const error = conversationError || messagesError || updateError;
    if (!error) return null;

    return (
      <Alert variant="destructive" className="my-4">
        <AlertCircle className="w-4 h-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error instanceof Error
            ? error.message
            : "Failed to load conversation"}
          <div className="mt-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleRetry}
              className="flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </Button>
          </div>
        </AlertDescription>
      </Alert>
    );
  };

  // Improved Skeleton loader
  const MessageSkeleton = () => (
    <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
      {/* Welcome message skeleton */}
      <div className="flex flex-col items-center justify-center px-4 text-center opacity-70">
        <Skeleton className="w-16 h-16 mb-4 rounded-full" />
        <Skeleton className="w-48 mb-2 h-7" />
        <Skeleton className="w-64 h-4 mb-1" />
        <Skeleton className="w-56 h-4 mb-1" />
        <Skeleton className="h-4 w-60" />
      </div>

      {/* Message skeletons */}
      {[
        { role: "user", length: "short" },
        { role: "assistant", length: "long" },
        { role: "user", length: "medium" },
        { role: "assistant", length: "very-long" },
      ].map((msg, idx) => {
        const isUser = msg.role === "user";
        const lengths = {
          short: "w-[180px]",
          medium: "w-[250px]",
          long: "w-[320px]",
          "very-long": "w-[380px]",
        };

        return (
          <div
            key={idx}
            className={cn(
              "flex items-start gap-3",
              isUser ? "justify-end" : "justify-start",
            )}
          >
            {/* Avatar skeleton */}
            {!isUser && (
              <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/10">
                <Cpu className="w-4 h-4 text-primary/40" />
              </div>
            )}

            {/* Message bubble skeleton */}
            <div
              className={cn(
                "flex flex-col gap-1 p-3 rounded-2xl max-w-[90%]",
                isUser ? "bg-primary/20" : "bg-muted/80",
                lengths[msg.length as keyof typeof lengths],
              )}
            >
              {/* Message content lines */}
              <Skeleton className="w-full h-4" />
              <Skeleton className="h-4 w-[90%]" />
              {(msg.length === "long" || msg.length === "very-long") && (
                <>
                  <Skeleton className="h-4 w-[80%]" />
                  <Skeleton className="h-4 w-[85%]" />
                </>
              )}
              {msg.length === "very-long" && (
                <>
                  <Skeleton className="h-4 w-[70%]" />
                  <Skeleton className="h-4 w-[75%]" />
                </>
              )}
              {/* Time stamp skeleton */}
              <Skeleton className="self-end w-12 h-3 mt-1 opacity-50" />
            </div>

            {/* User avatar skeleton */}
            {isUser && (
              <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/20">
                <User className="w-4 h-4 text-primary/40" />
              </div>
            )}
          </div>
        );
      })}

      {/* Typing indicator skeleton */}
      <div className="flex items-start gap-3">
        <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/10">
          <Cpu className="w-4 h-4 text-primary/40" />
        </div>
        <div className="flex items-center gap-1 p-3 rounded-2xl bg-muted/80 w-[100px]">
          <Skeleton className="w-2 h-2 rounded-full" />
          <Skeleton className="w-2 h-2 rounded-full" />
          <Skeleton className="w-2 h-2 rounded-full" />
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col w-full h-full min-h-full bg-background text-foreground">
      <ChatHeader
        currentModel={currentModel}
        currentProvider={currentProvider}
        resetConversation={resetConversation}
        title={title}
        setTitle={setTitle}
      />
      {initialLoading ? (
        <MessageSkeleton />
      ) : getErrorContent() ? (
        getErrorContent()
      ) : (
        <MessageList
          conversationId={numericConversationId}
          messages={messages}
          isLoading={isSendingMessage}
          error={sendError ? String(sendError) : null}
          isStreaming={isStreaming}
          streamingContent={streamingContent}
        />
      )}
      <ChatFooter
        isLoading={isLoading}
        sendMessage={sendMessage}
        showActions={showActions}
        toggleActions={toggleActions}
        isStreaming={isStreaming}
        abortStreaming={abortStreaming}
      />
    </div>
  );
}
