import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteMessage } from "@/services/messages";
import { DBMessage } from "@/services/messages/types";
import { useDeleteMessages } from "./use-delete-messages";
import { useAuth } from "@clerk/clerk-react";

export function useDeleteMessage() {
  const queryClient = useQueryClient();
  const { mutateAsync: deleteMessages } = useDeleteMessages();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation({
    mutationFn: async ({
      messageId,
      messages,
      index,
      conversationId,
    }: {
      conversationId: number;
      messageId: number;
      messages: DBMessage[];
      index: number;
    }) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      const hasSubsequentMessages = index < messages.length - 1;

      if (hasSubsequentMessages) {
        const subsequentMessageIds = messages
          .slice(index + 1)
          .map((msg) => msg.id);

        await deleteMessages({
          messageIds: subsequentMessageIds,
          conversationId,
        });
      }

      return deleteMessage(messageId, token);
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
}
