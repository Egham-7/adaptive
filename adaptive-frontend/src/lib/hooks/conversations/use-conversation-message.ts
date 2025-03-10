import { useQuery } from "@tanstack/react-query";
import { getMessages } from "@/services/messages";

export const useConversationMessages = (conversationId: number) =>
  useQuery({
    queryKey: ["messages", conversationId],
    queryFn: async () => {
      const dbMessages = await getMessages(conversationId);

      return dbMessages;
    },
  });
