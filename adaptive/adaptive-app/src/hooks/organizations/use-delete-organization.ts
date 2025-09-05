import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useDeleteOrganization = () => {
	const utils = api.useUtils();

	return api.organizations.delete.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.organizations.getAll.cancel();

			// Snapshot the previous value
			const previousOrganizations = utils.organizations.getAll.getData();

			// Optimistically update to the new value
			utils.organizations.getAll.setData(undefined, (old) => {
				if (!old) return old;
				return old.filter((org) => org.id !== variables.id);
			});

			// Return context object with the snapshotted value
			return { previousOrganizations };
		},
		onSuccess: () => {
			toast.success("Organization deleted successfully!");
			// Invalidate and refetch related queries
			utils.organizations.getAll.invalidate();
		},
		onError: (error, _variables, context) => {
			toast.error(error.message || "Failed to delete organization");
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousOrganizations) {
				utils.organizations.getAll.setData(
					undefined,
					context.previousOrganizations,
				);
			}
		},
		onSettled: () => {
			// Always refetch after error or success to ensure consistency
			utils.organizations.getAll.invalidate();
		},
	});
};
