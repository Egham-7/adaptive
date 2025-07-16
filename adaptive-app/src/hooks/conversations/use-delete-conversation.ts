import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useDeleteConversation = () => {
	const utils = api.useUtils();

	return api.conversations.delete.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.conversations.list.cancel();

			// Snapshot the previous value
			const previousConversations = utils.conversations.list.getData();

			// Optimistically update to the new value
			utils.conversations.list.setData(undefined, (old) => {
				if (!old) return old;
				return old.filter((conversation) => conversation.id !== variables.id);
			});

			// Return context object with the snapshotted value
			return { previousConversations };
		},
		onSuccess: () => {
			utils.conversations.list.invalidate();
		},
		onError: (error, _variables, context) => {
			toast.error(error.message);
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousConversations) {
				utils.conversations.list.setData(
					undefined,
					context.previousConversations,
				);
			}
		},
		onSettled: () => {
			// Always refetch after error or success to ensure consistency
			utils.conversations.list.invalidate();
		},
	});
};
