import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useCreateApiKey = () => {
	const utils = api.useUtils();

	return api.api_keys.create.useMutation({
		onSuccess: (data) => {
			toast.success("API key created successfully!");
			// Invalidate and refetch the list
			utils.api_keys.list.invalidate();
			return data;
		},
		onError: (error) => {
			toast.error(error.message || "Failed to create API key");
		},
	});
};
