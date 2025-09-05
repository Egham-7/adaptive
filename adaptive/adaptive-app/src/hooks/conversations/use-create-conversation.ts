import { useRouter } from "next/navigation";
import { api } from "@/trpc/react";

export const useCreateConversation = () => {
	const utils = api.useUtils();
	const router = useRouter();

	return api.conversations.create.useMutation({
		onSuccess: (newConversation) => {
			// Invalidate the list to refetch and show the new item
			utils.conversations.list.invalidate();
			// Navigate to the newly created chat
			router.push(`/chat-platform/chats/${newConversation.id}`);
		},
	});
};
