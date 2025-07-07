import { api } from "@/trpc/react";

export const useProjects = (organizationId: string) => {
	return api.projects.getByOrganization.useQuery(
		{ organizationId },
		{
			staleTime: 5 * 60 * 1000, // 5 minutes
			refetchOnWindowFocus: false,
			enabled: !!organizationId,
		},
	);
};
