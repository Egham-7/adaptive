import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useDeleteOrganization = () => {
	const utils = api.useUtils();

	return api.organizations.delete.useMutation({
		onSuccess: () => {
			toast.success("Organization deleted successfully!");
			// Invalidate and refetch related queries
			utils.organizations.getAll.invalidate();
		},
		onError: (error) => {
			toast.error(error.message || "Failed to delete organization");
		},
	});
};
