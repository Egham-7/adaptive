import { useCallback, useEffect, useMemo, useReducer } from "react";
import { Edit3, RotateCcw, ThumbsDown, ThumbsUp, Trash2 } from "lucide-react";
import type { UIMessage } from "@ai-sdk/react";

import { useDeleteMessage } from "@/hooks/messages/use-delete-message";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { MessageInput } from "./message-input";
import { MessageList } from "@/components/ui/chat/message-list";
import { PromptSuggestions } from "./prompt-suggestions";
import SubscribeButton from "@/app/_components/stripe/subscribe-button";
import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";
import { cn } from "@/lib/utils";

import { ChatContainer } from "./chat-container";
import { ChatErrorDisplay } from "./chat-error-display";
import { ChatForm } from "./chat-form";
import { ChatMessages } from "./chat-messages";
import { useOptimisticMessageCount } from "./chat-hooks";
import { messageReducer } from "./chat-reducer";
import type { ChatProps, MessageTextPart } from "./chat-types";
import { getMessageContent } from "./chat-utils";

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
}: ChatProps) {
  const deleteMessageMutation = useDeleteMessage();
  const [state, dispatch] = useReducer(messageReducer, {
    messages,
    editingMessageId: null,
    editingContent: "",
  });

  const optimisticRemainingMessages = useOptimisticMessageCount(
    messages,
    remainingMessages,
  );

  // Sync external messages with internal state
  useEffect(() => {
    dispatch({ type: "SET_MESSAGES", messages });
  }, [messages]);

  // Computed values
  const displayRemainingMessages =
    optimisticRemainingMessages ?? remainingMessages;
  const usedMessages =
    displayRemainingMessages !== undefined
      ? DAILY_MESSAGE_LIMIT - displayRemainingMessages
      : 0;

  const lastMessage = state.messages.at(-1);
  const isEmpty = state.messages.length === 0;
  const isTyping = lastMessage?.role === "user";

  // Event handlers
  const handleStop = useCallback(() => {
    stop?.();
    const lastAssistantMessage = state.messages.findLast(
      (m) => m.role === "assistant",
    );
    if (lastAssistantMessage) {
      dispatch({
        type: "CANCEL_TOOL_INVOCATIONS",
        messageId: lastAssistantMessage.id,
      });
    }
  }, [stop, state.messages]);

  const handleEditMessage = useCallback(
    (messageId: string, content: string) => {
      dispatch({ type: "EDIT_MESSAGE", messageId, content });
    },
    [],
  );

  const handleSaveEdit = useCallback(
    (messageId: string) => {
      if (!state.editingContent.trim()) return;

      const messageIndex = state.messages.findIndex((m) => m.id === messageId);
      if (messageIndex === -1) return;

      // Delete subsequent messages
      const messagesToDelete = state.messages.slice(messageIndex);
      messagesToDelete.forEach((msg) => {
        deleteMessageMutation.mutate({ id: msg.id });
      });

      setMessages(messages.slice(0, messageIndex));
      sendMessage?.({ text: state.editingContent.trim() });
      dispatch({ type: "CLEAR_EDITING" });
    },
    [
      state.editingContent,
      state.messages,
      deleteMessageMutation,
      setMessages,
      messages,
      sendMessage,
    ],
  );

  const handleCancelEdit = useCallback(() => {
    dispatch({ type: "CLEAR_EDITING" });
  }, []);

  const handleRetryMessage = useCallback(
    (message: UIMessage) => {
      if (!sendMessage) return;

      const messageIndex = state.messages.findIndex((m) => m.id === message.id);
      if (messageIndex !== -1) {
        console.log("Retrying message:", message.id);
        const messagesToDelete = state.messages.slice(0, messageIndex + 1);
        console.log(
          "Messages to delete:",
          messagesToDelete.map((m) => m.id),
        );
        messagesToDelete.forEach((msg) => {
          deleteMessageMutation.mutate({ id: msg.id });
        });
      }
      setMessages(messages.slice(0, messageIndex));
      dispatch({ type: "RETRY_MESSAGE", messageId: message.id });
      sendMessage({ text: getMessageContent(message) });
    },
    [sendMessage, state.messages, deleteMessageMutation],
  );

  const handleDeleteMessage = useCallback(
    (messageId: string) => {
      const messageIndex = state.messages.findIndex((m) => m.id === messageId);
      if (messageIndex !== -1) {
        const messagesToDelete = state.messages.slice(messageIndex);
        messagesToDelete.forEach((msg) => {
          deleteMessageMutation.mutate({ id: msg.id });
        });
      }
      setMessages(messages.slice(0, messageIndex));
      dispatch({ type: "DELETE_MESSAGE_AND_AFTER", messageId });
    },
    [state.messages, deleteMessageMutation],
  );

  const handleRateMessage = useCallback(
    (messageId: string, rating: "thumbs-up" | "thumbs-down") => {
      onRateResponse?.(messageId, rating);
    },
    [onRateResponse],
  );

  // Message options factory
  const messageOptions = useCallback(
    (message: UIMessage) => {
      const isUserMessage = message.role === "user";
      const canEdit = isUserMessage && !isGenerating;
      const canRetry = isUserMessage && !isGenerating;
      const canDelete = !isGenerating;

      if (isUserMessage) {
        return {
          actions: (
            <>
              {canEdit && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() =>
                    handleEditMessage(message.id, getMessageContent(message))
                  }
                  disabled={state.editingMessageId === message.id}
                >
                  <Edit3 className="h-4 w-4" />
                </Button>
              )}
              {canRetry && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => handleRetryMessage(message)}
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              {canDelete && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 text-destructive hover:text-destructive"
                  onClick={() => handleDeleteMessage(message.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </>
          ),
          isEditing: state.editingMessageId === message.id,
          editingContent: state.editingContent,
          onEditingContentChange: (content: string) =>
            dispatch({
              type: "EDIT_MESSAGE",
              messageId: state.editingMessageId || "",
              content,
            }),
          onSaveEdit: () => handleSaveEdit(message.id),
          onCancelEdit: handleCancelEdit,
        };
      }

      // Assistant message actions
      const messageContent =
        (message.parts?.find((p) => p.type === "text") as MessageTextPart)
          ?.text || getMessageContent(message);

      // Check if this is the latest assistant message and we're generating
      const isLatestAssistantMessage = 
        message.id === lastMessage?.id && 
        lastMessage?.role === "assistant";
      const shouldStream = isGenerating && isLatestAssistantMessage;

      return {
        enableStreaming: shouldStream,
        streamingMode: "typewriter" as const,
        streamingSpeed: 40,
        actions: onRateResponse ? (
          <>
            <div className="border-r pr-1">
              <CopyButton
                content={messageContent}
                copyMessage="Copied response to clipboard!"
              />
            </div>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => handleRateMessage(message.id, "thumbs-up")}
            >
              <ThumbsUp className="h-4 w-4" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => handleRateMessage(message.id, "thumbs-down")}
            >
              <ThumbsDown className="h-4 w-4" />
            </Button>
          </>
        ) : (
          <CopyButton
            content={messageContent}
            copyMessage="Copied response to clipboard!"
          />
        ),
      };
    },
    [
      onRateResponse,
      isGenerating,
      state.editingMessageId,
      state.editingContent,
      handleEditMessage,
      handleSaveEdit,
      handleCancelEdit,
      handleRetryMessage,
      handleDeleteMessage,
      handleRateMessage,
    ],
  );

  // Render message counter
  const MessageCounter = useMemo(() => {
    if (
      limitsLoading ||
      isUnlimited ||
      displayRemainingMessages === undefined
    ) {
      return null;
    }

    return (
      <div className="mx-4 mb-2 text-center">
        <span
          className={cn(
            "text-xs px-2 py-1 rounded-full",
            hasReachedLimit
              ? "bg-red-100 text-red-700"
              : displayRemainingMessages <= 2
                ? "bg-orange-100 text-orange-700"
                : "bg-gray-100 text-gray-600",
          )}
        >
          {usedMessages}/{DAILY_MESSAGE_LIMIT} messages used today
        </span>
      </div>
    );
  }, [
    limitsLoading,
    isUnlimited,
    displayRemainingMessages,
    hasReachedLimit,
    usedMessages,
  ]);

  // Render message limit warning
  const MessageLimitWarning = useMemo(() => {
    if (
      limitsLoading ||
      isUnlimited ||
      displayRemainingMessages === undefined ||
      displayRemainingMessages > 5
    ) {
      return null;
    }

    return (
      <div className="mx-4 mt-4 text-center">
        <div className="inline-block rounded-lg border border-orange-200 bg-orange-50 p-4 text-left">
          <p className="text-sm text-orange-800">
            {displayRemainingMessages > 0
              ? `You have ${displayRemainingMessages} messages remaining today.`
              : "You've reached your daily message limit."}
            {userId && (
              <SubscribeButton
                userId={userId}
                variant="link"
                className="ml-1 text-orange-800 hover:text-orange-900"
              >
                Upgrade to Pro
              </SubscribeButton>
            )}{" "}
            for unlimited messages.
          </p>
        </div>
      </div>
    );
  }, [limitsLoading, isUnlimited, displayRemainingMessages, userId]);

  return (
    <ChatContainer className={className}>
      {isEmpty && sendMessage && suggestions && (
        <PromptSuggestions
          label="Try these prompts ✨"
          sendMessage={sendMessage}
          suggestions={suggestions}
          enableCategories={true}
        />
      )}

      {state.messages.length > 0 && (
        <ChatMessages messages={state.messages}>
          <MessageList
            messages={state.messages}
            isTyping={isTyping}
            messageOptions={messageOptions}
          />
        </ChatMessages>
      )}

      {isError && <ChatErrorDisplay error={error} onRetry={onRetry} />}

      {MessageCounter}
      {MessageLimitWarning}

      <ChatForm
        className="mt-auto"
        isPending={isGenerating || isTyping}
        handleSubmit={handleSubmit}
        hasReachedLimit={hasReachedLimit}
      >
        {({ files, setFiles }) => (
          <MessageInput
            value={input}
            onChange={handleInputChange}
            allowAttachments
            files={files}
            setFiles={setFiles}
            stop={handleStop}
            isGenerating={isGenerating}
            transcribeAudio={transcribeAudio}
            disabled={hasReachedLimit}
            enableAdvancedFeatures={true}
            placeholder={
              hasReachedLimit
                ? "Daily message limit reached - upgrade to continue"
                : "Ask me anything..."
            }
          />
        )}
      </ChatForm>
    </ChatContainer>
  );
}

