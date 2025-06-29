import { useCallback, useEffect, useReducer, useRef, useState, useMemo } from "react";
import type { UIMessage } from "@ai-sdk/react";
import { messageReducer } from "./chat-reducer";
import type { MessageState } from "./chat-types";
import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";

export function useOptimisticMessageCount(
  messages: UIMessage[],
  remainingMessages?: number,
) {
  const [optimisticRemainingMessages, setOptimisticRemainingMessages] =
    useState(remainingMessages);
  const messagesRef = useRef(messages);

  useEffect(() => {
    setOptimisticRemainingMessages(remainingMessages);
  }, [remainingMessages]);

  useEffect(() => {
    if (
      messagesRef.current.length < messages.length &&
      remainingMessages !== undefined
    ) {
      const newUserMessages = messages.filter((m) => m.role === "user").length;
      const oldUserMessages = messagesRef.current.filter(
        (m) => m.role === "user",
      ).length;

      if (newUserMessages > oldUserMessages) {
        setOptimisticRemainingMessages((prev) => Math.max(0, (prev || 0) - 1));
      }
    }
    messagesRef.current = messages;
  }, [messages, remainingMessages]);

  return optimisticRemainingMessages;
}

/**
 * Hook for managing chat message state with editing capabilities
 */
export function useChatState(initialMessages: UIMessage[]) {
  const [state, dispatch] = useReducer(messageReducer, {
    messages: initialMessages,
    editingMessageId: null,
    editingContent: "",
  });

  // Sync external messages with internal state
  useEffect(() => {
    dispatch({ type: "SET_MESSAGES", messages: initialMessages });
  }, [initialMessages]);

  const startEditing = useCallback((messageId: string, content: string) => {
    dispatch({ type: "EDIT_MESSAGE", messageId, content });
  }, []);

  const updateEditingContent = useCallback((messageId: string, content: string) => {
    dispatch({ type: "EDIT_MESSAGE", messageId, content });
  }, []);

  const clearEditing = useCallback(() => {
    dispatch({ type: "CLEAR_EDITING" });
  }, []);

  const cancelToolInvocations = useCallback((messageId: string) => {
    dispatch({ type: "CANCEL_TOOL_INVOCATIONS", messageId });
  }, []);

  const deleteMessageAndAfter = useCallback((messageId: string) => {
    dispatch({ type: "DELETE_MESSAGE_AND_AFTER", messageId });
  }, []);

  const retryMessage = useCallback((messageId: string) => {
    dispatch({ type: "RETRY_MESSAGE", messageId });
  }, []);

  // Computed values
  const lastMessage = state.messages.at(-1);
  const isEmpty = state.messages.length === 0;

  return {
    state,
    actions: {
      startEditing,
      updateEditingContent,
      clearEditing,
      cancelToolInvocations,
      deleteMessageAndAfter,
      retryMessage,
    },
    computed: {
      lastMessage,
      isEmpty,
    },
  };
}

/**
 * Hook for managing chat actions (edit, delete, retry, etc.)
 */
export function useChatActions({
  chatState,
  chatActions,
  messages,
  setMessages,
  sendMessage,
  deleteMessageMutation,
  isGenerating,
}: {
  chatState: MessageState;
  chatActions: ReturnType<typeof useChatState>['actions'];
  messages: UIMessage[];
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  sendMessage?: (message: { text: string }) => void;
  deleteMessageMutation: { mutate: (params: { id: string }) => void };
  isGenerating: boolean;
}) {
  const handleEditMessage = useCallback(
    (messageId: string, content: string) => {
      chatActions.startEditing(messageId, content);
    },
    [chatActions],
  );

  const handleSaveEdit = useCallback(
    (messageId: string) => {
      if (!chatState.editingContent.trim()) return;

      const messageIndex = chatState.messages.findIndex((m) => m.id === messageId);
      if (messageIndex === -1) return;

      // Delete subsequent messages
      const messagesToDelete = chatState.messages.slice(messageIndex);
      messagesToDelete.forEach((msg) => {
        deleteMessageMutation.mutate({ id: msg.id });
      });

      setMessages(messages.slice(0, messageIndex));
      sendMessage?.({ text: chatState.editingContent.trim() });
      chatActions.clearEditing();
    },
    [
      chatState.editingContent,
      chatState.messages,
      deleteMessageMutation,
      setMessages,
      messages,
      sendMessage,
      chatActions,
    ],
  );

  const handleCancelEdit = useCallback(() => {
    chatActions.clearEditing();
  }, [chatActions]);

  const handleRetryMessage = useCallback(
    (message: UIMessage) => {
      if (!sendMessage) return;

      const messageIndex = chatState.messages.findIndex((m) => m.id === message.id);
      if (messageIndex !== -1) {
        const messagesToDelete = chatState.messages.slice(0, messageIndex + 1);
        messagesToDelete.forEach((msg) => {
          deleteMessageMutation.mutate({ id: msg.id });
        });
      }
      setMessages(messages.slice(0, messageIndex));
      chatActions.retryMessage(message.id);
      
      // Get message content
      const textPart = message.parts?.find((p) => p.type === "text") as
        | { text: string }
        | undefined;
      const content = textPart?.text || "";
      
      sendMessage({ text: content });
    },
    [sendMessage, chatState.messages, deleteMessageMutation, messages, setMessages, chatActions],
  );

  const handleDeleteMessage = useCallback(
    (messageId: string) => {
      const messageIndex = chatState.messages.findIndex((m) => m.id === messageId);
      if (messageIndex !== -1) {
        const messagesToDelete = chatState.messages.slice(messageIndex);
        messagesToDelete.forEach((msg) => {
          deleteMessageMutation.mutate({ id: msg.id });
        });
      }
      setMessages(messages.slice(0, messageIndex));
      chatActions.deleteMessageAndAfter(messageId);
    },
    [chatState.messages, deleteMessageMutation, messages, setMessages, chatActions],
  );

  const handleStop = useCallback(() => {
    const lastAssistantMessage = chatState.messages.findLast(
      (m) => m.role === "assistant",
    );
    if (lastAssistantMessage) {
      chatActions.cancelToolInvocations(lastAssistantMessage.id);
    }
  }, [chatState.messages, chatActions]);

  // Determine message capabilities
  const getMessageCapabilities = useCallback(
    (message: UIMessage) => {
      const isUserMessage = message.role === "user";
      return {
        canEdit: isUserMessage && !isGenerating,
        canRetry: isUserMessage && !isGenerating,
        canDelete: !isGenerating,
        isEditing: chatState.editingMessageId === message.id,
      };
    },
    [isGenerating, chatState.editingMessageId],
  );

  return {
    handleEditMessage,
    handleSaveEdit,
    handleCancelEdit,
    handleRetryMessage,
    handleDeleteMessage,
    handleStop,
    getMessageCapabilities,
  };
}

/**
 * Hook for managing chat limits and message counting
 */
export function useChatLimits({
  messages,
  remainingMessages,
  hasReachedLimit = false,
  isUnlimited = true,
  limitsLoading = false,
}: {
  messages: UIMessage[];
  remainingMessages?: number;
  hasReachedLimit?: boolean;
  isUnlimited?: boolean;
  limitsLoading?: boolean;
}) {
  const optimisticRemainingMessages = useOptimisticMessageCount(
    messages,
    remainingMessages,
  );

  const displayRemainingMessages = optimisticRemainingMessages ?? remainingMessages;
  
  const usedMessages = useMemo(
    () =>
      displayRemainingMessages !== undefined
        ? DAILY_MESSAGE_LIMIT - displayRemainingMessages
        : 0,
    [displayRemainingMessages],
  );

  const shouldShowCounter = useMemo(
    () => !limitsLoading && !isUnlimited && displayRemainingMessages !== undefined,
    [limitsLoading, isUnlimited, displayRemainingMessages],
  );

  const shouldShowWarning = useMemo(
    () =>
      shouldShowCounter &&
      displayRemainingMessages !== undefined &&
      displayRemainingMessages <= 5,
    [shouldShowCounter, displayRemainingMessages],
  );

  const limitStatus = useMemo((): "normal" | "low" | "warning" | "reached" => {
    if (hasReachedLimit) return "reached";
    if (displayRemainingMessages !== undefined && displayRemainingMessages <= 2) return "warning";
    if (displayRemainingMessages !== undefined && displayRemainingMessages <= 5) return "low";
    return "normal";
  }, [hasReachedLimit, displayRemainingMessages]);

  return {
    displayRemainingMessages,
    usedMessages,
    shouldShowCounter,
    shouldShowWarning,
    limitStatus,
    hasReachedLimit,
    isUnlimited,
    limitsLoading,
  };
}

/**
 * Hook for managing message rating functionality
 */
export function useChatRating({
  onRateResponse,
}: {
  onRateResponse?: (messageId: string, rating: "thumbs-up" | "thumbs-down") => void;
}) {
  const handleRateMessage = useCallback(
    (messageId: string, rating: "thumbs-up" | "thumbs-down") => {
      onRateResponse?.(messageId, rating);
    },
    [onRateResponse],
  );

  return {
    handleRateMessage,
    canRate: !!onRateResponse,
  };
}