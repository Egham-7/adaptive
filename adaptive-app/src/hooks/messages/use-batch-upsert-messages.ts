import { api } from "@/trpc/react";

export const useBatchUpsertMessages = () => {
	const utils = api.useUtils();

	return api.messages.batchUpsert.useMutation({
		onSuccess: (_result, variables) => {
			// Invalidate and refetch conversation messages after batch operation
			utils.messages.listByConversation.invalidate({
				conversationId: variables.conversationId,
			});
		},
	});
};
