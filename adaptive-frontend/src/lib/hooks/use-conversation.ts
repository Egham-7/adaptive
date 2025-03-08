import { useState } from "react";
import { useConversationState } from "./use-conversation-state";
import { useConversationData } from "./use-conversation-data";
import { useSendMessage } from "./use-send-message";
import { useUpdateConversation } from "./use-update-conversation";
import { ChatCompletionResponse } from "@/services/llms/types";

interface UseConversationOptions {
  conversationId: number;
  initialTitle?: string;
}

/**
 * Hook for managing a single conversation with storage capabilities
 */
export const useConversation = (options: UseConversationOptions) => {
  const { conversationId } = options;
  const { title, setTitle: setStateTitle } = useConversationState(options);

  const {
    messages,
    isLoading: isLoadingData,
    error: dataError,
  } = useConversationData(conversationId);

  const [lastResponse, setLastResponse] =
    useState<ChatCompletionResponse | null>(null);

  const updateConversationMutation = useUpdateConversation();

  const {
    sendMessage: originalSendMessage,
    isLoading: isSendingMessage,
    error: sendError,
  } = useSendMessage(conversationId, messages);

  // Wrap setTitle to also update the database
  const setTitle = (newTitle: string) => {
    setStateTitle(newTitle);

    // Only update in DB if we have a conversation ID
    if (conversationId) {
      updateConversationMutation.mutate({
        id: conversationId,
        title: newTitle,
      });
    }
  };

  const sendMessage = async (content: string) => {
    const response = await originalSendMessage(content);
    setLastResponse(response.response);
    return response;
  };

  const resetConversation = () => {
    setLastResponse(null);
  };

  // Combined loading and error states
  const isLoading =
    isLoadingData || isSendingMessage || updateConversationMutation.isPending;
  const error = dataError || sendError || updateConversationMutation.error;

  return {
    // Data
    messages,
    conversationId,
    title,
    lastResponse,
    // Actions
    sendMessage,
    setTitle,
    resetConversation,
    // Status
    isLoading,
    error: error ? String(error) : null,
  };
};
