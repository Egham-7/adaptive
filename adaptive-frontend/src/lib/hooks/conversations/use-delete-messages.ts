import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteMessages } from "@/services/messages";

export const useDeleteMessages = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      messageIds,
    }: {
      messageIds: number[];
      conversationId: number;
    }) => deleteMessages(messageIds),
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
};
