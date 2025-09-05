import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useUpdateOrganization = () => {
	const utils = api.useUtils();

	return api.organizations.update.useMutation({
		onMutate: async (variables) => {
			// Cancel any outgoing refetches
			await utils.organizations.getAll.cancel();
			await utils.organizations.getById.cancel({ id: variables.id });

			// Snapshot the previous values
			const previousOrganizations = utils.organizations.getAll.getData();
			const previousOrganization = utils.organizations.getById.getData({
				id: variables.id,
			});

			// Optimistically update to the new values
			if (previousOrganization) {
				const optimisticOrganization = {
					...previousOrganization,
					...variables,
				};
				utils.organizations.getById.setData(
					{ id: variables.id },
					optimisticOrganization,
				);
			}

			utils.organizations.getAll.setData(undefined, (old) => {
				if (!old) return old;
				return old.map((org) =>
					org.id === variables.id ? { ...org, ...variables } : org,
				);
			});

			// Return context object with the snapshotted values
			return { previousOrganizations, previousOrganization };
		},
		onSuccess: (_data, variables) => {
			toast.success("Organization updated successfully!");
			// Invalidate and refetch related queries
			utils.organizations.getAll.invalidate();
			utils.organizations.getById.invalidate({ id: variables.id });
		},
		onError: (error, variables, context) => {
			toast.error(error.message || "Failed to update organization");
			// If the mutation fails, use the context returned from onMutate to roll back
			if (context?.previousOrganizations) {
				utils.organizations.getAll.setData(
					undefined,
					context.previousOrganizations,
				);
			}
			if (context?.previousOrganization) {
				utils.organizations.getById.setData(
					{ id: variables.id },
					context.previousOrganization,
				);
			}
		},
		onSettled: (_data, _error, variables) => {
			// Always refetch after error or success to ensure consistency
			utils.organizations.getAll.invalidate();
			utils.organizations.getById.invalidate({ id: variables.id });
		},
	});
};
