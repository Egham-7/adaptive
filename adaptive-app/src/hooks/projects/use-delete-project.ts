import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useDeleteProject = () => {
	const utils = api.useUtils();

	return api.projects.delete.useMutation({
		onSuccess: () => {
			toast.success("Project deleted successfully!");
			// Invalidate and refetch related queries
			utils.projects.getByOrganization.invalidate();
			utils.organizations.getAll.invalidate();
		},
		onError: (error) => {
			toast.error(error.message || "Failed to delete project");
		},
	});
};
