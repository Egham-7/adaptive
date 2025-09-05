import { api } from "@/trpc/react";

export const useOrganizations = () => {
	return api.organizations.getAll.useQuery(undefined, {
		staleTime: 5 * 60 * 1000, // 5 minutes
		refetchOnWindowFocus: false,
	});
};
