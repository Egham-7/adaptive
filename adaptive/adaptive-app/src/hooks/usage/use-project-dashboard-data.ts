"use client";

import { useProjectAnalytics } from "@/hooks/usage/use-project-analytics";
import type { DashboardFilters } from "@/types/api-platform/dashboard";

export function useProjectDashboardData(
	projectId: string,
	filters: DashboardFilters,
) {
	const {
		data: analyticsData,
		isLoading,
		error,
		refetch,
	} = useProjectAnalytics({
		projectId,
		startDate: filters.dateRange?.from,
		endDate: filters.dateRange?.to,
		provider: filters.provider === "all" ? undefined : filters.provider,
	});

	return {
		data: analyticsData,
		loading: isLoading,
		error: error?.message || null,
		refresh: refetch,
	};
}
