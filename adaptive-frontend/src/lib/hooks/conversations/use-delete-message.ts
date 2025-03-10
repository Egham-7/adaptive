import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteMessage } from "@/services/messages";
import { DBMessage } from "@/services/messages/types";
import { useDeleteMessages } from "./use-delete-messages";

export function useDeleteMessage() {
  const queryClient = useQueryClient();
  const { mutateAsync: deleteMessages } = useDeleteMessages();

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

      return deleteMessage(messageId);
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
