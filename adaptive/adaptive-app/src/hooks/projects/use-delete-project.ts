import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useDeleteProject = () => {
	const utils = api.useUtils();

	return api.projects.delete.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.projects.getByOrganization.cancel();

			// Get the project to delete for the organizationId
			const projectToDelete = utils.projects.getById?.getData?.({
				id: variables.id,
			});

			if (!projectToDelete) return;

			// Snapshot the previous value
			const previousProjects = utils.projects.getByOrganization.getData({
				organizationId: projectToDelete.organizationId,
			});

			// Optimistically update to the new value
			utils.projects.getByOrganization.setData(
				{ organizationId: projectToDelete.organizationId },
				(oldData) => {
					if (!oldData) return oldData;
					return oldData.filter((project) => project.id !== variables.id);
				},
			);

			// Return context object with the snapshotted value
			return {
				previousProjects,
				organizationId: projectToDelete.organizationId,
			};
		},
		onSuccess: () => {
			toast.success("Project deleted successfully!");
			// Invalidate and refetch related queries
			utils.projects.getByOrganization.invalidate();
			utils.organizations.getAll.invalidate();
		},
		onError: (error, _variables, context) => {
			toast.error(error.message || "Failed to delete project");
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousProjects && context.organizationId) {
				utils.projects.getByOrganization.setData(
					{ organizationId: context.organizationId },
					context.previousProjects,
				);
			}
		},
		onSettled: (_data, _error, _variables) => {
			// Always refetch after error or success to ensure consistency
			utils.projects.getByOrganization.invalidate();
			utils.organizations.getAll.invalidate();
		},
	});
};
