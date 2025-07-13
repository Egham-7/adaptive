import { useCallback } from "react";
import type { UIMessage } from "@ai-sdk/react";
import type { MessageState } from "../chat-types";
import { getMessageText, findMessageIndex } from "../utils/message-utils";

interface MessageActionsHookProps {
  messageState: MessageState;
  externalMessages: UIMessage[];
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  sendMessage?: (message: { text: string }) => Promise<void>;
  deleteMessageMutation: { mutate: (params: { id: string }) => void };
  isGenerating: boolean;
  onClearEditing: () => void;
  onRetryMessage: (messageId: string) => void;
  onDeleteMessageAndAfter: (messageId: string) => void;
}

/**
 * Hook for message actions (edit, delete, retry, etc.)
 * Provides clean, type-safe action handlers
 */
export function useMessageActions({
  messageState,
  externalMessages,
  setMessages,
  sendMessage,
  deleteMessageMutation,
  onClearEditing,
  onRetryMessage,
  onDeleteMessageAndAfter,
}: MessageActionsHookProps) {
  const handleSaveEdit = useCallback(
    (messageId: string) => {
      if (!messageState.editingContent.trim()) return;

      const messageIndex = findMessageIndex(externalMessages, messageId);
      if (messageIndex === -1) return;

      // Delete subsequent messages from database
      const messagesToDelete = externalMessages.slice(messageIndex);
      messagesToDelete.forEach((msg) => {
        deleteMessageMutation.mutate({ id: msg.id });
      });

      // Keep messages up to the edited one, then send new message
      const messagesBeforeEdit = externalMessages.slice(0, messageIndex);
      setMessages(messagesBeforeEdit);
      sendMessage?.({ text: messageState.editingContent.trim() });
      onClearEditing();
    },
    [
      messageState.editingContent,
      deleteMessageMutation,
      setMessages,
      externalMessages,
      sendMessage,
      onClearEditing,
    ],
  );

  const handleRetryMessage = useCallback(
    (message: UIMessage) => {
      if (!sendMessage) return;

      const messageIndex = findMessageIndex(externalMessages, message.id);
      if (messageIndex === -1) return;

      // Delete this message and all subsequent messages from database
      const messagesToDelete = externalMessages.slice(messageIndex);
      messagesToDelete.forEach((msg) => {
        deleteMessageMutation.mutate({ id: msg.id });
      });

      // Keep messages before the one being retried
      const messagesBeforeRetry = externalMessages.slice(0, messageIndex);
      setMessages(messagesBeforeRetry);
      onRetryMessage(message.id);

      const content = getMessageText(message);
      sendMessage({ text: content });
    },
    [
      sendMessage,
      deleteMessageMutation,
      externalMessages,
      setMessages,
      onRetryMessage,
    ],
  );

  const handleDeleteMessage = useCallback(
    (messageId: string) => {
      const messageIndex = findMessageIndex(externalMessages, messageId);
      if (messageIndex === -1) return;

      const messagesToDelete = externalMessages.slice(messageIndex);
      messagesToDelete.forEach((msg) => {
        deleteMessageMutation.mutate({ id: msg.id });
      });

      setMessages(externalMessages.slice(0, messageIndex));
      onDeleteMessageAndAfter(messageId);
    },
    [
      deleteMessageMutation,
      externalMessages,
      setMessages,
      onDeleteMessageAndAfter,
    ],
  );

  const handleStop = useCallback(() => {
    const lastAssistantMessage = messageState.messages.findLast(
      (m) => m.role === "assistant",
    );
    if (lastAssistantMessage) {
      // Handle stop logic here
    }
  }, [messageState.messages]);

  return {
    handleSaveEdit,
    handleRetryMessage,
    handleDeleteMessage,
    handleStop,
  };
}

