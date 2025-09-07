import { api } from "@/trpc/react";

export const useDeleteMessage = () => {
	const utils = api.useUtils();

	return api.messages.delete.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.messages.listByConversation.cancel();

			// Get the message to delete for the conversationId
			const messageToDelete = utils.messages.getById.getData({
				id: variables.id,
			});

			if (!messageToDelete) return;

			// Snapshot the previous value
			const previousMessages = utils.messages.listByConversation.getData({
				conversationId: messageToDelete.conversationId,
			});

			// Optimistically update to the new value
			utils.messages.listByConversation.setData(
				{ conversationId: messageToDelete.conversationId },
				(oldData) => {
					if (!oldData) return oldData;
					return oldData.filter((msg) => msg.id !== variables.id);
				},
			);

			// Return context object with the snapshotted value
			return {
				previousMessages,
				conversationId: messageToDelete.conversationId,
			};
		},
		onSuccess: (_deletedMessage, variables, context) => {
			// Remove from conversation messages cache using context if available
			if (context?.conversationId) {
				utils.messages.listByConversation.setData(
					{ conversationId: context.conversationId },
					(oldData) => {
						if (!oldData) return oldData;
						return oldData.filter((msg) => msg.id !== variables.id);
					},
				);
			}

			// Invalidate the specific message query
			utils.messages.getById.invalidate({ id: variables.id });
		},
		onError: (_err, _variables, context) => {
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousMessages && context.conversationId) {
				utils.messages.listByConversation.setData(
					{ conversationId: context.conversationId },
					context.previousMessages,
				);
			}
		},
		onSettled: (_data, _error, _variables, context) => {
			// Always refetch after error or success to ensure consistency
			if (context?.conversationId) {
				utils.messages.listByConversation.invalidate({
					conversationId: context.conversationId,
				});
			}
		},
	});
};
