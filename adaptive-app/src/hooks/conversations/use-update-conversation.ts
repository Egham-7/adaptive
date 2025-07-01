import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useUpdateConversation = () => {
	const utils = api.useUtils();

	return api.conversations.update.useMutation({
		onSuccess: () => {
			utils.conversations.list.invalidate();
		},
		onError: (error) => {
			toast.error(error.message);
		},
	});
};
