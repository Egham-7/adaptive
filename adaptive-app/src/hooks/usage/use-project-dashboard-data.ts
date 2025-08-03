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
			singleProvider:
				trend.spend +
				analyticsData.totalSavings *
					(trend.spend / analyticsData.totalSpend || 0), // Proportional alternative cost
			requests: trend.requests,
			spend: trend.spend,
			tokens: trend.tokens,
			inputTokens: trend.inputTokens, // ← Add input tokens from backend
			outputTokens: trend.outputTokens, // ← Add output tokens from backend
		}));

		// Transform daily trends to token data
		const tokenData = analyticsData.dailyTrends.map((trend) => ({
			date: new Date(trend.date).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			tokens: trend.tokens,
			spend: trend.spend,
			requests: trend.requests,
		}));

		// Transform daily trends to request data
		const requestData = analyticsData.dailyTrends.map((trend) => ({
			date: new Date(trend.date).toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			requests: trend.requests,
			spend: trend.spend,
			tokens: trend.tokens,
		}));

		// Use actual savings from API
		const totalSavings = analyticsData.totalSavings;
		const savingsPercentage = analyticsData.totalSavingsPercentage;

		// Transform request type breakdown to task breakdown
		const taskBreakdown = analyticsData.requestTypeBreakdown.map(
			(requestType, index) => {
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
					comparisonCost: `$${(requestType.spend + analyticsData.totalSavings * (requestType.spend / analyticsData.totalSpend || 0)).toFixed(2)}`,
					savings: `$${(analyticsData.totalSavings * (requestType.spend / analyticsData.totalSpend || 0)).toFixed(2)}`,
					percentage: analyticsData.totalSavingsPercentage || 0,
				};
			},
		);

		// Transform provider breakdown to providers - API now returns all providers with correct calculations
		const providers = analyticsData.providerBreakdown.map((provider) => {
			return {
				id: provider.provider,
				name:
					provider.provider.charAt(0).toUpperCase() +
					provider.provider.slice(1),
				icon:
					PROVIDER_ICONS[provider.provider as keyof typeof PROVIDER_ICONS] ||
					"/logos/default.svg",
				comparisonCosts: {
					adaptive: analyticsData.totalSpend, // Actual total cost with Adaptive
					single: provider.estimatedSingleProviderCost, // What ALL requests would cost on this provider
				},
			};
		});

		const errorRateData = analyticsData.dailyTrends.map((trend) => ({
			date: trend.date.toISOString().split("T")[0] || "",
			errorRate:
				trend.requests > 0 ? (trend.errorCount / trend.requests) * 100 : 0,
			errorCount: trend.errorCount,
		}));

		return {
			totalSpend: analyticsData.totalSpend,
			totalSavings,
			savingsPercentage,
			totalTokens: analyticsData.totalTokens,
			totalRequests: analyticsData.totalRequests,
			errorRate: analyticsData.errorRate,
			errorCount: analyticsData.errorCount,
			usageData,
			tokenData,
			requestData,
			errorRateData,
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
