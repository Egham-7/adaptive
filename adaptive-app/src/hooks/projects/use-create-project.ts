import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useCreateProject = () => {
	const utils = api.useUtils();

	return api.projects.create.useMutation({
		onSuccess: () => {
			toast.success("Project created successfully!");
			// Invalidate and refetch related queries
			utils.projects.getByOrganization.invalidate();
			utils.organizations.getAll.invalidate();
		},
		onError: (error) => {
			toast.error(error.message || "Failed to create project");
		},
	});
};
