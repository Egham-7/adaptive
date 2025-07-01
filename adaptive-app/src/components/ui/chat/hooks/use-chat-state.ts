import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";
import type { ChatLimitsProps, ChatErrorProps, ChatRatingProps } from "../chat-types";
import { useMessageState } from "./use-message-state";
import { useMessageSync } from "./use-message-sync";
import { useMessageCapabilities } from "./use-message-capabilities";
import { useMessageActions } from "./use-message-actions";
import { useChatLimits } from "./use-chat-limits";
import { shouldShowStreaming } from "../utils/message-utils";

interface ChatStateHookProps {
  initialMessages: UIMessage[];
  messages: UIMessage[];
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  sendMessage?: (message: { text: string }) => void;
  deleteMessageMutation: { mutate: (params: { id: string }) => void };
  isGenerating: boolean;
  stop?: () => void;
  limitsProps: ChatLimitsProps;
  errorProps: ChatErrorProps;
  ratingProps: ChatRatingProps;
}

/**
 * Main chat state management hook
 * Orchestrates all chat-related state and provides clean interface
 */
export function useChatState({
  initialMessages,
  messages,
  setMessages,
  sendMessage,
  deleteMessageMutation,
  isGenerating,
  stop,
  limitsProps,
  errorProps,
  ratingProps,
}: ChatStateHookProps) {
  
  // Core message state
  const messageState = useMessageState(initialMessages);
  
  // Sync external messages with internal state
  useMessageSync(
    initialMessages,
    messageState.state.messages,
    messageState.actions.setMessages
  );

  // Chat limits state
  const limitsState = useChatLimits(messages, limitsProps);

  // Error state
  const errorState = {
    isError: errorProps.isError || false,
    error: errorProps.error,
    canRetry: !!errorProps.onRetry,
  };

  // Rating state
  const ratingState = {
    canRate: !!ratingProps.onRateResponse,
  };

  // Message capabilities
  const capabilities = useMessageCapabilities({
    isGenerating,
    editingMessageId: messageState.state.editingMessageId,
    canRate: ratingState.canRate,
  });

  // Message actions
  const messageActions = useMessageActions({
    messageState: messageState.state,
    externalMessages: messages,
    setMessages,
    sendMessage,
    deleteMessageMutation,
    isGenerating,
    onClearEditing: messageState.actions.clearEditing,
    onRetryMessage: messageState.actions.retryMessage,
    onDeleteMessageAndAfter: messageState.actions.deleteMessageAndAfter,
  });

  // Enhanced stop handler
  const handleStop = useCallback(() => {
    stop?.();
    if (messageState.computed.lastAssistantMessage) {
      messageState.actions.cancelToolInvocations(
        messageState.computed.lastAssistantMessage.id
      );
    }
  }, [stop, messageState.actions, messageState.computed.lastAssistantMessage]);

  // Message options factory with clean streaming logic
  const createMessageOptions = useCallback((message: UIMessage) => {
    const messageCaps = capabilities.getMessageCapabilities(message);
    const isStreaming = shouldShowStreaming(message, messages, isGenerating);

    return {
      capabilities: messageCaps,
      isStreaming,
      editingContent: messageState.state.editingContent,
      onEditingContentChange: (content: string) =>
        messageState.actions.updateEditingContent(message.id, content),
      onSaveEdit: () => messageActions.handleSaveEdit(message.id),
      onCancelEdit: messageState.actions.clearEditing,
      onRetry: () => messageActions.handleRetryMessage(message),
      onDelete: () => messageActions.handleDeleteMessage(message.id),
      onRate: ratingProps.onRateResponse 
        ? (messageId: string, rating: "thumbs-up" | "thumbs-down") => ratingProps.onRateResponse!(messageId, rating)
        : undefined,
    };
  }, [
    capabilities,
    messages,
    isGenerating,
    messageState,
    messageActions,
    ratingProps.onRateResponse,
  ]);

  return {
    // State
    messages: messageState.state.messages,
    computed: messageState.computed,
    limits: limitsState,
    error: errorState,
    rating: ratingState,
    
    // Actions
    messageActions: {
      startEditing: messageState.actions.startEditing,
      clearEditing: messageState.actions.clearEditing,
      ...messageActions,
    },
    
    // Utilities
    createMessageOptions,
    handleStop,
    
    // Computed values
    isTyping: messageState.computed.lastMessage?.role === "user" && !errorState.isError,
  };
}