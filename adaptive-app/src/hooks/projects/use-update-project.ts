import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useUpdateProject = () => {
	const utils = api.useUtils();

	return api.projects.update.useMutation({
		onSuccess: (_data, variables) => {
			toast.success("Project updated successfully!");
			// Invalidate and refetch related queries
			utils.projects.getByOrganization.invalidate();
			utils.projects.getById.invalidate({ id: variables.id });
			utils.organizations.getAll.invalidate();
		},
		onError: (error) => {
			toast.error(error.message || "Failed to update project");
		},
	});
};
