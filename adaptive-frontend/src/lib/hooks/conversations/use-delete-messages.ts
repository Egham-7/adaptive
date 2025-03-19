import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteMessages } from "@/services/messages";
import { useAuth } from "@clerk/clerk-react";

export const useDeleteMessages = () => {
  const queryClient = useQueryClient();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation({
    mutationFn: async ({
      messageIds,
    }: {
      messageIds: number[];
      conversationId: number;
    }) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      return deleteMessages(messageIds, token);
    },

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
