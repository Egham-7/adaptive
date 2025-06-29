import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";

import { useDeleteMessage } from "@/hooks/messages/use-delete-message";
import { MessageList } from "@/components/ui/chat/message-list";
import { cn } from "@/lib/utils";

import { ChatContainer } from "./chat-container";
import { ChatErrorDisplay } from "./chat-error-display";
import { ChatMessages } from "./chat-messages";
import { ErrorDisplay } from "./error-display";
import { MessageInputWrapper } from "./message-input-wrapper";
import { WelcomeScreen } from "./welcome-screen";
import { ChatStatus } from "./chat-status";
import { MessageActions } from "./message-actions";
import {
  useChatState,
  useChatActions,
  useChatLimits,
  useChatRating,
} from "./chat-hooks";
import type { ChatProps } from "./chat-types";

export function Chat({
  messages,
  handleSubmit,
  input,
  handleInputChange,
  stop,
  isGenerating,
  sendMessage,
  suggestions,
  className,
  onRateResponse,
  setMessages,
  transcribeAudio,
  isError,
  error,
  onRetry,
  hasReachedLimit = false,
  remainingMessages,
  isUnlimited = true,
  limitsLoading = false,
  userId,
  showWelcomeInterface = false,
}: ChatProps) {
  const deleteMessageMutation = useDeleteMessage();

  // Initialize all hooks
  const chatState = useChatState(messages);
  const chatLimits = useChatLimits({
    messages,
    remainingMessages,
    hasReachedLimit,
    isUnlimited,
    limitsLoading,
  });
  const chatRating = useChatRating({ onRateResponse });
  const chatActions = useChatActions({
    chatState: chatState.state,
    chatActions: chatState.actions,
    messages,
    setMessages,
    sendMessage,
    deleteMessageMutation,
    isGenerating,
  });

  // Computed values
  const isTyping = chatState.computed.lastMessage?.role === "user" && !isError;

  // Event handlers
  const handleStop = useCallback(() => {
    stop?.();
    chatActions.handleStop();
  }, [stop, chatActions]);

  // Message options factory
  const messageOptions = useCallback(
    (message: UIMessage) => {
      const capabilities = chatActions.getMessageCapabilities(message);

      return {
        actions: (
          <MessageActions
            message={message}
            canEdit={capabilities.canEdit}
            canRetry={capabilities.canRetry}
            canDelete={capabilities.canDelete}
            canRate={chatRating.canRate}
            isEditing={capabilities.isEditing}
            onEdit={chatActions.handleEditMessage}
            onRetry={chatActions.handleRetryMessage}
            onDelete={chatActions.handleDeleteMessage}
            onRate={chatRating.handleRateMessage}
          />
        ),
        isEditing: capabilities.isEditing,
        editingContent: chatState.state.editingContent,
        onEditingContentChange: (content: string) =>
          chatState.actions.updateEditingContent(message.id, content),
        onSaveEdit: () => chatActions.handleSaveEdit(message.id),
        onCancelEdit: chatActions.handleCancelEdit,
        isStreaming: isGenerating,
        isError,
        error,
        onRetryError: onRetry,
      };
    },
    [chatActions, chatRating, chatState, isGenerating, isError, error, onRetry],
  );

  // Common chat status props
  const chatStatusProps = {
    shouldShowCounter: chatLimits.shouldShowCounter,
    shouldShowWarning: chatLimits.shouldShowWarning,
    limitStatus: chatLimits.limitStatus,
    usedMessages: chatLimits.usedMessages,
    displayRemainingMessages: chatLimits.displayRemainingMessages,
    userId,
  };

  // Show welcome screen if configured and no messages
  if (showWelcomeInterface && chatState.computed.isEmpty && sendMessage) {
    return (
      <WelcomeScreen
        className={className}
        suggestions={suggestions ?? []}
        sendMessage={sendMessage}
        handleSubmit={handleSubmit}
        input={input}
        handleInputChange={handleInputChange}
        isGenerating={isGenerating}
        isTyping={isTyping}
        hasReachedLimit={hasReachedLimit}
        transcribeAudio={transcribeAudio}
        isError={isError}
        error={error}
        onRetry={onRetry}
        onStop={handleStop}
        {...chatStatusProps}
      />
    );
  }

  // Main chat interface
  return (
    <ChatContainer className={cn("h-full", className)}>
      {chatState.state.messages.length > 0 && (
        <ChatMessages
          messages={chatState.state.messages}
          isStreaming={isGenerating}
        >
          <MessageList
            messages={chatState.state.messages}
            isTyping={isTyping}
            messageOptions={messageOptions}
          />
        </ChatMessages>
      )}

      {/* Error feedback */}
      {error && <ChatErrorDisplay />}
      <ErrorDisplay isError={isError} error={error} onRetry={onRetry} />

      <ChatStatus {...chatStatusProps} />

      <MessageInputWrapper
        className="mt-auto mb-6"
        isPending={isGenerating || isTyping}
        handleSubmit={handleSubmit}
        hasReachedLimit={hasReachedLimit}
        value={input}
        onChange={handleInputChange}
        stop={handleStop}
        isGenerating={isGenerating}
        transcribeAudio={transcribeAudio}
      />
    </ChatContainer>
  );
}
