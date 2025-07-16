import { api } from "@/trpc/react";

export const useDeleteMessage = () => {
	const utils = api.useUtils();

	return api.messages.delete.useMutation({
		onSuccess: (deletedMessage, variables) => {
			// Remove from conversation messages cache
			utils.messages.listByConversation.setData(
				{ conversationId: deletedMessage.conversationId },
				(oldData) => {
					if (!oldData) return oldData;
					return oldData.filter((msg) => msg.id !== variables.id);
				},
			);

			// Invalidate the specific message query
			utils.messages.getById.invalidate({ id: variables.id });
		},
	});
};
