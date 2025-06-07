import { api } from "@/trpc/react";
import type { Message } from "prisma/generated"; // Assuming Message comes from prisma/generated, adjust if different

export const useCreateMessage = () => {
  const utils = api.useUtils();

  return api.messages.create.useMutation({
    onSuccess: (newMessage, variables) => {
      utils.messages.listByConversation.invalidate({
        conversationId: variables.conversationId,
      });

      utils.messages.listByConversation.setData(
        { conversationId: variables.conversationId },
        (oldData: Message[] | undefined) => {
          if (!oldData) {
            // If no old data, return the new message in an array
            return [newMessage];
          }
          // If old data exists, spread it and add the new message
          return [...oldData, newMessage];
        },
      );
    },
  });
};
