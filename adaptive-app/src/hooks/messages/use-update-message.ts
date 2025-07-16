import { api } from "@/trpc/react";

export const useUpdateMessage = () => {
	const utils = api.useUtils();

	return api.messages.update.useMutation({
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
	});
};
