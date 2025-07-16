import { api } from "@/trpc/react";
import type { Message } from "@/types";

export const useUpdateMessage = () => {
	const utils = api.useUtils();

	return api.messages.update.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.messages.listByConversation.cancel();
			await utils.messages.getById.cancel({ id: variables.id });

			// Snapshot the previous values
			const previousMessage = utils.messages.getById.getData({
				id: variables.id,
			});
			const previousMessages = previousMessage
				? utils.messages.listByConversation.getData({
						conversationId: previousMessage.conversationId,
					})
				: undefined;

			if (!previousMessage) return;

			// Optimistically update the specific message
			const optimisticMessage = {
				...previousMessage,
				...variables,
				metadata: variables.metadata ?? previousMessage.metadata,
				annotations: variables.annotations ?? previousMessage.annotations,
				parts: variables.parts ?? previousMessage.parts,
			} as Message;
			utils.messages.getById.setData({ id: variables.id }, optimisticMessage);

			// Update the message in the conversation list cache
			utils.messages.listByConversation.setData(
				{ conversationId: previousMessage.conversationId },
				(oldData) => {
					if (!oldData) return oldData;
					return oldData.map((msg) =>
						msg.id === variables.id ? optimisticMessage : msg,
					);
				},
			);

			// Return context object with the snapshotted values
			return {
				previousMessage,
				previousMessages,
				conversationId: previousMessage.conversationId,
			};
		},
		onSuccess: (updatedMessage, variables) => {
			// Update the specific message in cache
			utils.messages.getById.setData({ id: variables.id }, updatedMessage);

			// Update the message in the conversation list cache
			utils.messages.listByConversation.setData(
				{ conversationId: updatedMessage.conversationId },
				(oldData) => {
					if (!oldData) return oldData;
					return oldData.map((msg) =>
						msg.id === variables.id ? updatedMessage : msg,
					);
				},
			);
		},
		onError: (_err, variables, context) => {
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousMessage) {
				utils.messages.getById.setData(
					{ id: variables.id },
					context.previousMessage,
				);
			}
			if (context?.previousMessages && context.conversationId) {
				utils.messages.listByConversation.setData(
					{ conversationId: context.conversationId },
					context.previousMessages,
				);
			}
		},
		onSettled: (data, _error, variables) => {
			// Always refetch after error or success to ensure consistency
			utils.messages.getById.invalidate({ id: variables.id });
			if (data?.conversationId) {
				utils.messages.listByConversation.invalidate({
					conversationId: data.conversationId,
				});
			}
		},
	});
};
