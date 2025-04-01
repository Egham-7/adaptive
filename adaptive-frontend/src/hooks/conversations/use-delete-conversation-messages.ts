import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAllMessages } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";

/**
 * Hook for deleting all messages in a conversation
 *
 * @returns A mutation object for deleting all messages in a conversation
 */
export const useDeleteConversationMessages = () => {
  const queryClient = useQueryClient();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation({
    mutationFn: async (conversationId: number) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      return deleteAllMessages(conversationId, token);
    },
    onSuccess: (_, conversationId) => {
      queryClient.invalidateQueries({
        queryKey: ["messages", conversationId],
      });
    },
  });
};
