import { toast } from "sonner";
import { api } from "@/trpc/react";

export const useCreateOrganization = () => {
	const utils = api.useUtils();

	return api.organizations.create.useMutation({
		onSuccess: () => {
			toast.success("Organization created successfully!");
			// Invalidate and refetch the organizations list
			utils.organizations.getAll.invalidate();
		},
		onError: (error) => {
			toast.error(error.message || "Failed to create organization");
		},
	});
};
