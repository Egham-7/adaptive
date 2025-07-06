import { api } from "@/trpc/react";

export const useOrganization = (organizationId: string) => {
	return api.organizations.getById.useQuery(
		{ id: organizationId },
		{
			staleTime: 5 * 60 * 1000, // 5 minutes
			refetchOnWindowFocus: false,
			enabled: !!organizationId,
		},
	);
};
