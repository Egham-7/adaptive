import { useQuery } from "@tanstack/react-query";
import { getConversation } from "@/services/conversations";
import { useConversationMesages } from "./use-conversation-message";
import { convertToApiMessages } from "@/services/messages";

export const useConversationData = (conversationId?: number) => {
  // Query conversation data if ID exists
  const {
    data: conversation,
    isLoading: isLoadingConversation,
    error: conversationError,
  } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => (conversationId ? getConversation(conversationId) : null),
    enabled: !!conversationId,
  });

  const {
    data: dbMessages = [],
    isLoading: isLoadingMessages,
    error: messagesError,
  } = useConversationMesages(conversationId);

  // Convert DB messages to API format
  const messages = convertToApiMessages(dbMessages);

  return {
    conversation,
    messages,
    isLoading: isLoadingConversation || isLoadingMessages,
    error: conversationError || messagesError,
  };
};
