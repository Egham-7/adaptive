import { api } from "@/trpc/react";

export const useMessagesByConversation = (conversationId: number) => {
	return api.messages.listByConversation.useQuery(
		{ conversationId },
		{
			enabled: !!conversationId,
		},
	);
};
