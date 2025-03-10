import { useQuery } from "@tanstack/react-query";
import { getConversation } from "@/services/conversations";

export const useConversation = (conversationId: number) => {
  return useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId),
  });
};
