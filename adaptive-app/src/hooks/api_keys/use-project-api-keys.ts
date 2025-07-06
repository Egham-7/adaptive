import { api } from "@/trpc/react";

export const useProjectApiKeys = (projectId: string) => {
	return api.api_keys.getByProject.useQuery(
		{ projectId },
		{
			staleTime: 5 * 60 * 1000, // 5 minutes
			refetchOnWindowFocus: false,
			enabled: !!projectId,
		},
	);
};
