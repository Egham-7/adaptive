import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useUpdateOrganization = () => {
	const utils = api.useUtils();

	return api.organizations.update.useMutation({
		onSuccess: (_data, variables) => {
			toast.success("Organization updated successfully!");
			// Invalidate and refetch related queries
			utils.organizations.getAll.invalidate();
			utils.organizations.getById.invalidate({ id: variables.id });
		},
		onError: (error) => {
			toast.error(error.message || "Failed to update organization");
		},
	});
};
