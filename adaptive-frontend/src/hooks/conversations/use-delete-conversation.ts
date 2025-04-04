import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteConversation } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";
import { toast } from "sonner";
/**
 * Hook for deleting a conversation
 *
 * @returns A mutation object for deleting a conversation
 */
export const useDeleteConversation = () => {
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

      return deleteConversation(conversationId, token);
    },
    onSuccess: () => {
      // Invalidate the conversations list query when a conversation is deleted
      queryClient.invalidateQueries({
        queryKey: ["conversations"],
      });
      toast.success("Conversation deleted successfully.");
    },
    onError: () => {
      toast.error("Failed to delete conversation.");
    },
  });
};
