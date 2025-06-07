import { api } from "@/trpc/react";
import { toast } from "sonner";

export const useDeleteApiKey = () => {
	const utils = api.useUtils();

	return api.api_keys.delete.useMutation({
		onSuccess: (_, variables) => {
			toast.success("API key deleted successfully!");
			// Invalidate the list and remove the specific item from cache
			utils.api_keys.list.invalidate();
			utils.api_keys.getById.invalidate({ id: variables.id });
		},
		onError: (error) => {
			toast.error(error.message || "Failed to delete API key");
		},
	});
};
