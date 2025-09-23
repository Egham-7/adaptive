import { api } from "@/trpc/react";
import type { ProviderType } from "@/types/api-platform/dashboard";

interface ProjectAnalyticsParams {
	projectId: string;
	startDate?: Date;
	endDate?: Date;
	provider?: ProviderType;
}

export const useProjectAnalytics = ({
	projectId,
	startDate,
	endDate,
	provider,
}: ProjectAnalyticsParams) => {
	const query = api.projectAnalytics.getProjectAnalytics.useQuery(
		{
			projectId,
			startDate,
			endDate,
			provider,
		},
		{
			staleTime: 5 * 60 * 1000, // 5 minutes
			refetchOnWindowFocus: false,
			enabled: !!projectId,
		},
	);

	return query;
};
