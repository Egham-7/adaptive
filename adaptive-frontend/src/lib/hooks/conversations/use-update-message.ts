import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateMessage } from "@/services/messages";

export function useUpdateMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      conversationId,
      messageId,
      updates,
    }: {
      conversationId: number;
      messageId: number;
      updates: { role?: string; content?: string };
    }) => updateMessage(conversationId, messageId, updates),
    
    onSuccess: (_updatedMessage, variables) => {
      // Invalidate and refetch conversations queries
      queryClient.invalidateQueries({
        queryKey: ["conversations", variables.conversationId],
      });

      // Invalidate message list for the conversation we are in
      
      queryClient.invalidateQueries({
        queryKey: ["messages", variables.conversationId],
      });
      
     ;
    },
  });
}