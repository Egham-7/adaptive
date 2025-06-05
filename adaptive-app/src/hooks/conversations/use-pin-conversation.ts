import { api } from "@/trpc/react";

export const usePinConversation = () => {
  const utils = api.useUtils();

  return api.conversations.setPinStatus.useMutation({
    onMutate: async (variables) => {
      await utils.conversations.list.cancel();
      const previousConversations = utils.conversations.list.getData();
      utils.conversations.list.setData(undefined, (old) =>
        old?.map((c) =>
          c.id === variables.id ? { ...c, pinned: variables.pinned } : c,
        ),
      );
      return { previousConversations };
    },
    onError: (err, _newTodo, context) => {
      utils.conversations.list.setData(
        undefined,
        context?.previousConversations,
      );
    },
    onSettled: () => {
      utils.conversations.list.invalidate();
    },
  });
};
