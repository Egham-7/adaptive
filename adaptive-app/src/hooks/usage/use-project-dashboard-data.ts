"use client";

import { useMemo } from "react";
import { useProjectAnalytics } from "@/hooks/usage/use-project-analytics";
import type {
	DashboardData,
	DashboardFilters,
} from "@/types/api-platform/dashboard";

const PROVIDER_ICONS = {
	openai: "/logos/openai.webp",
	anthropic: "/logos/anthropic.jpeg",
	google: "/logos/google.svg",
	groq: "/logos/groq.png",
	deepseek: "/logos/deepseek.svg",
	huggingface: "/logos/huggingface.png",
} as const;

const formatNumber = (num: number): string => {
	if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
	if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
	return num.toString();
};

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

	const transformedData = useMemo((): DashboardData | null => {
		if (!analyticsData) return null;

		// Transform daily trends to usage data
		const usageData = analyticsData.dailyTrends.map((trend) => ({
			date: new Date(trend.date).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			adaptive: trend.spend,
			singleProvider: trend.spend * 2.3, // Estimated single provider cost
			requests: trend.requests,
		}));

		// Transform daily trends to token data
		const tokenData = analyticsData.dailyTrends.map((trend) => ({
			date: new Date(trend.date).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			tokens: trend.tokens,
		}));

		// Transform daily trends to request data
		const requestData = analyticsData.dailyTrends.map((trend) => ({
			date: new Date(trend.date).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			requests: trend.requests,
		}));

		// Calculate savings based on estimated single provider costs
		const estimatedSingleProviderCost = analyticsData.totalSpend * 2.3;
		const totalSavings = estimatedSingleProviderCost - analyticsData.totalSpend;
		const savingsPercentage =
			estimatedSingleProviderCost > 0
				? (totalSavings / estimatedSingleProviderCost) * 100
				: 0;

		// Transform request type breakdown to task breakdown
		const taskBreakdown = analyticsData.requestTypeBreakdown.map(
			(requestType, index) => {
				const estimatedSingleCost = requestType.spend * 2.3;
				const savings = estimatedSingleCost - requestType.spend;
				const savingsPercentage =
					estimatedSingleCost > 0 ? (savings / estimatedSingleCost) * 100 : 0;

				return {
					id: index.toString(),
					name:
						requestType.type.charAt(0).toUpperCase() +
						requestType.type.slice(1),
					icon: `/icons/${requestType.type}.svg`, // Fallback icon
					requests: formatNumber(requestType.requests),
					inputTokens: formatNumber(Math.floor(requestType.tokens * 0.6)), // Estimated input tokens
					outputTokens: formatNumber(Math.floor(requestType.tokens * 0.4)), // Estimated output tokens
					cost: `$${requestType.spend.toFixed(2)}`,
					comparisonCost: `$${estimatedSingleCost.toFixed(2)}`,
					savings: `$${savings.toFixed(2)}`,
					percentage: savingsPercentage,
				};
			},
		);

		// Transform provider breakdown to providers
		const providers = analyticsData.providerBreakdown.map((provider) => {
			const estimatedSingleCost = provider.spend * 2.3;

			return {
				id: provider.provider,
				name:
					provider.provider.charAt(0).toUpperCase() +
					provider.provider.slice(1),
				icon:
					PROVIDER_ICONS[provider.provider as keyof typeof PROVIDER_ICONS] ||
					"/logos/default.svg",
				comparisonCosts: {
					adaptive: provider.spend,
					single: estimatedSingleCost,
				},
			};
		});

		return {
			totalSpend: analyticsData.totalSpend,
			totalSavings,
			savingsPercentage,
			totalTokens: analyticsData.totalTokens,
			totalRequests: analyticsData.totalRequests,
			usageData,
			tokenData,
			requestData,
			taskBreakdown,
			providers,
		};
	}, [analyticsData]);

	return {
		data: transformedData,
		loading: isLoading,
		error: error?.message || null,
		refresh: refetch,
	};
}
