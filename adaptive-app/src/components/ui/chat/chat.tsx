import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";

import { useDeleteMessage } from "@/hooks/messages/use-delete-message";
import { MessageList } from "@/components/ui/chat/message-list";
import { cn } from "@/lib/shared/utils";

import { ChatContainer } from "./chat-container";
import { ChatErrorDisplay } from "./chat-error-display";
import { ChatMessages } from "./chat-messages";
import { ErrorDisplay } from "./error-display";
import { MessageInputWrapper } from "./message-input-wrapper";
import { WelcomeScreen } from "./welcome-screen";
import { ChatStatus } from "./chat-status";
import { MessageActions } from "./message-actions";
import { useChatState } from "./hooks/use-chat-state";
import type { ChatProps } from "./chat-types";

export function Chat({
  messages,
  handleSubmit,
  handleSuggestionSubmit,
  sendMessage,
  input,
  handleInputChange,
  stop,
  isGenerating,
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

  // Initialize unified chat state
  const chatState = useChatState({
    initialMessages: messages as UIMessage[],
    messages: messages as UIMessage[],
    setMessages: setMessages as React.Dispatch<
      React.SetStateAction<UIMessage[]>
    >,
    deleteMessageMutation,
    isGenerating,
    stop,
    sendMessage,
    limitsProps: {
      hasReachedLimit,
      remainingMessages,
      isUnlimited,
      limitsLoading,
    },
    errorProps: {
      isError,
      error,
      onRetry,
    },
    ratingProps: {
      onRateResponse,
    },
  });

  // Message options factory with clean interface
  const messageOptions = useCallback(
    (message: UIMessage) => {
      const options = chatState.createMessageOptions(message);

      return {
        actions: (
          <MessageActions
            message={message}
            canEdit={options.capabilities.canEdit}
            canRetry={options.capabilities.canRetry}
            canDelete={options.capabilities.canDelete}
            canRate={options.capabilities.canRate}
            isEditing={options.capabilities.isEditing}
            onEdit={chatState.messageActions.startEditing}
            onRetry={options.onRetry}
            onDelete={options.onDelete}
            onRate={options.onRate}
          />
        ),
        isEditing: options.capabilities.isEditing,
        editingContent: options.editingContent,
        onEditingContentChange: options.onEditingContentChange,
        onSaveEdit: options.onSaveEdit,
        onCancelEdit: options.onCancelEdit,
        isStreaming: options.isStreaming,
        isError,
        error,
        onRetryError: onRetry,
      };
    },
    [chatState, isError, error, onRetry],
  );

  // Common chat status props
  const chatStatusProps = {
    shouldShowCounter: chatState.limits.shouldShowCounter,
    shouldShowWarning: chatState.limits.shouldShowWarning,
    limitStatus: chatState.limits.limitStatus,
    usedMessages: chatState.limits.usedMessages,
    displayRemainingMessages: chatState.limits.displayRemainingMessages,
    userId,
  };

  // Show welcome screen if configured and no messages
  if (showWelcomeInterface && chatState.computed.isEmpty) {
    return (
      <WelcomeScreen
        className={className}
        suggestions={suggestions ?? []}
        onSuggestionClick={
          handleSuggestionSubmit ||
          ((text: string) => {
            handleInputChange({
              target: { value: text },
            } as React.ChangeEvent<HTMLTextAreaElement>);
            handleSubmit();
          })
        }
        handleSubmit={handleSubmit}
        input={input}
        handleInputChange={handleInputChange}
        isGenerating={isGenerating}
        isTyping={chatState.isTyping}
        hasReachedLimit={chatState.limits.hasReachedLimit}
        transcribeAudio={transcribeAudio}
        isError={chatState.error.isError}
        error={chatState.error.error}
        onRetry={onRetry}
        onStop={chatState.handleStop}
        {...chatStatusProps}
      />
    );
  }

  // Main chat interface
  return (
    <ChatContainer className={cn("h-full relative", className)}>
      {/* Messages container with proper scroll handling */}
      <div className="absolute inset-0 flex flex-col">
        <div className="flex-1 overflow-hidden">
          {chatState.messages.length > 0 && (
            <ChatMessages
              messages={chatState.messages}
              isStreaming={isGenerating}
            >
              <MessageList
                messages={chatState.messages}
                messageOptions={messageOptions}
                isTyping={chatState.isTyping}
              />
            </ChatMessages>
          )}

          {/* Error feedback */}
          {chatState.error.error && <ChatErrorDisplay />}
          <ErrorDisplay
            isError={chatState.error.isError}
            error={chatState.error.error}
            onRetry={onRetry}
          />

          <ChatStatus {...chatStatusProps} />
        </div>

        {/* Fixed input at bottom */}
        <div className="flex-shrink-0 p-4 bg-background">
          <MessageInputWrapper
            className="mx-auto max-w-3xl"
            isPending={isGenerating || chatState.isTyping}
            handleSubmit={handleSubmit}
            hasReachedLimit={chatState.limits.hasReachedLimit}
            value={input}
            onChange={handleInputChange}
            stop={chatState.handleStop}
            isGenerating={isGenerating}
            transcribeAudio={transcribeAudio}
          />
        </div>
      </div>
    </ChatContainer>
  );
}
