import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteConversation } from "@/services/conversations";

/**
 * Hook for deleting a conversation
 *
 * @returns A mutation object for deleting a conversation
 */
export const useDeleteConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (conversationId: number) => {
      return deleteConversation(conversationId);
    },
    onSuccess: () => {
      // Invalidate the conversations list query when a conversation is deleted
      queryClient.invalidateQueries({
        queryKey: ["conversations"],
      });
    },
  });
};
