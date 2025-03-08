import { useQuery } from "@tanstack/react-query";
import { getMessages } from "@/services/messages";

export const useConversationMesages = (conversationId?: number) =>
  useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => (conversationId ? getMessages(conversationId) : []),
    enabled: !!conversationId,
  });
