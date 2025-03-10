import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteMessage } from "@/services/messages";

export function useDeleteMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      conversationId,
      messageId,
    }: {
      conversationId: number;
      messageId: number;
    }) => deleteMessage(conversationId, messageId),

    onSuccess: (_data, variables) => {
      // Invalidate and refetch conversations queries
      queryClient.invalidateQueries({
        queryKey: ["conversations", variables.conversationId],
      });

      queryClient.invalidateQueries({
        queryKey: ["messages", variables.conversationId],
      });
    },
  });
}
