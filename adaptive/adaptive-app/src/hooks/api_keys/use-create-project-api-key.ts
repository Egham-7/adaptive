import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useCreateProjectApiKey = () => {
	const utils = api.useUtils();

	return api.api_keys.createForProject.useMutation({
		onSuccess: (data, variables) => {
			toast.success("API key created successfully!");
			// Invalidate and refetch the project's API keys
			utils.api_keys.getByProject.invalidate({
				projectId: variables.projectId,
			});
			utils.api_keys.list.invalidate();
			return data;
		},
		onError: (error) => {
			toast.error(error.message || "Failed to create API key");
		},
	});
};
