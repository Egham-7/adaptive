import { useQuery } from "@tanstack/react-query";
import { convertToApiMessages, getMessages } from "@/services/messages";

export const useConversationMessages = (conversationId: number) =>
  useQuery({
    queryKey: ["messages", conversationId],
    queryFn: async () => {
      const dbMessages = await getMessages(conversationId);

      return convertToApiMessages(dbMessages);
    },
  });
