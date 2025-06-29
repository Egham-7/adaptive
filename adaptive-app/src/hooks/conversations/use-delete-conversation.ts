import { toast } from "sonner";
import { api } from "@/trpc/react";

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
