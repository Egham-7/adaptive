import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllMessages } from "@/services/conversations";

/**
 * Hook for deleting all messages in a conversation
 *
 * @returns A mutation object for deleting all messages in a conversation
 */
export const useDeleteConversationMessages = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (conversationId: number) => {
      return deleteAllMessages(conversationId);
    },
    onSuccess: (_, conversationId) => {
      queryClient.invalidateQueries({
        queryKey: ["messages", conversationId],
      });
    },
  });
};
