import { api } from "@/trpc/react";
import { toast } from "sonner";

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
