import { api } from "@/trpc/react";
import type { Message } from "@/types";

export const useCreateMessage = () => {
  const utils = api.useUtils();

  return api.messages.create.useMutation({
    onSuccess: (newMessage, variables) => {
      utils.messages.listByConversation.invalidate({
        conversationId: variables.conversationId,
      });

      utils.messages.listByConversation.setData(
        { conversationId: variables.conversationId },
        (oldData: Message) => {
          if (!oldData) return [newMessage];
          return [...oldData, newMessage];
        },
      a;
    },
  });
};
