import { api } from "@/trpc/react";
import { toast } from "sonner";

export const useDeleteConversation = () => {
	const utils = api.useUtils();

	return api.conversations.delete.useMutation({
		onSuccess: () => {
			utils.conversations.list.invalidate();
		},
		onError: (error) => {
			toast.error(error.message);
		},
	});
};
