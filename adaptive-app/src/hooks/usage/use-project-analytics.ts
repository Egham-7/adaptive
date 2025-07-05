import { api } from "@/trpc/react";

interface ProjectAnalyticsParams {
	projectId: string;
	startDate?: Date;
	endDate?: Date;
	provider?: string;
}

export const useProjectAnalytics = ({
	projectId,
	startDate,
	endDate,
	provider,
}: ProjectAnalyticsParams) => {
	return api.usage.getProjectAnalytics.useQuery(
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
};
