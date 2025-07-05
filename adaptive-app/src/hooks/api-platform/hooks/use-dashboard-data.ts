"use client";

import { useCallback, useEffect, useState } from "react";
import type {
	DashboardData,
	DashboardFilters,
} from "@/types/api-platform/dashboard";

const generateMockData = (filters: DashboardFilters): DashboardData => {
	const { dateRange } = filters;
	const daysDiff = Math.ceil(
		((dateRange?.to?.getTime() ?? 0) - (dateRange?.from?.getTime() ?? 0)) /
			(1000 * 60 * 60 * 24),
	);

	// Generate usage data
	const usageData = Array.from({ length: Math.min(daysDiff, 30) }, (_, i) => {
		const date = new Date(
			(dateRange?.from?.getTime() ?? 0) + i * 24 * 60 * 60 * 1000,
		);
		const adaptive = Math.random() * 0.3 + 0.1;
		const singleProvider = adaptive * (1.5 + Math.random() * 0.8);

		return {
			date: date.toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			adaptive: Number(adaptive.toFixed(2)),
			singleProvider: Number(singleProvider.toFixed(2)),
			requests: Math.floor(Math.random() * 200 + 50),
		};
	});

	// Generate token data
	const tokenData = Array.from({ length: Math.min(daysDiff, 30) }, (_, i) => {
		const date = new Date(
			(dateRange?.from?.getTime() ?? 0) + i * 24 * 60 * 60 * 1000,
		);
		return {
			date: date.toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			tokens: Math.floor(Math.random() * 20000 + 30000),
		};
	});

	// Generate request data
	const requestData = Array.from({ length: Math.min(daysDiff, 30) }, (_, i) => {
		const date = new Date(
			(dateRange?.from?.getTime() ?? 0) + i * 24 * 60 * 60 * 1000,
		);
		return {
			date: date.toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
			}),
			requests: Math.floor(Math.random() * 150 + 100),
		};
	});

	const totalSpend = usageData.reduce((sum, day) => sum + day.adaptive, 0);
	const totalComparison = usageData.reduce(
		(sum, day) => sum + day.singleProvider,
		0,
	);
	const totalSavings = totalComparison - totalSpend;
	const savingsPercentage = (totalSavings / totalComparison) * 100;

	const totalTokens = tokenData.reduce((sum, day) => sum + day.tokens, 0);
	const totalRequests = requestData.reduce((sum, day) => sum + day.requests, 0);

	const taskBreakdown = [
		{
			id: "1",
			name: "Code Generation",
			icon: "/icons/code.svg",
			requests: "2,847",
			inputTokens: "1.2M",
			outputTokens: "456K",
			cost: "$1.23",
			comparisonCost: "$2.89",
			savings: "$1.66",
			percentage: 42.6,
		},
		{
			id: "2",
			name: "Open Q&A",
			icon: "/icons/help-circle.svg",
			requests: "1,234",
			inputTokens: "890K",
			outputTokens: "234K",
			cost: "$0.67",
			comparisonCost: "$1.45",
			savings: "$0.78",
			percentage: 53.8,
		},
		{
			id: "3",
			name: "Summarization",
			icon: "/icons/file-text.svg",
			requests: "567",
			inputTokens: "234K",
			outputTokens: "89K",
			cost: "$0.12",
			comparisonCost: "$0.34",
			savings: "$0.22",
			percentage: 64.7,
		},
		{
			id: "4",
			name: "Translation",
			icon: "/icons/globe.svg",
			requests: "89",
			inputTokens: "45K",
			outputTokens: "52K",
			cost: "$0.45",
			comparisonCost: "$0.89",
			savings: "$0.44",
			percentage: 49.4,
		},
	];

	const providers = [
		{
			id: "openai",
			name: "OpenAI",
			icon: "/logos/openai.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 2.3 },
		},
		{
			id: "anthropic",
			name: "Anthropic",
			icon: "/logos/anthropic.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 1.98 },
		},
		{
			id: "gemini",
			name: "Gemini",
			icon: "/logos/google.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 1.71 },
		},
		{
			id: "cohere",
			name: "Cohere",
			icon: "/logos/cohere.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 1.61 },
		},
		{
			id: "grok",
			name: "Grok",
			icon: "/logos/grok.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 2.1 },
		},
		{
			id: "groq",
			name: "Groq",
			icon: "/logos/groq.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 1.85 },
		},
		{
			id: "deepseek",
			name: "Deepseek",
			icon: "/logos/deepseek.svg",
			comparisonCosts: { adaptive: totalSpend, single: totalSpend * 1.45 },
		},
	];

	return {
		totalSpend,
		totalSavings,
		savingsPercentage,
		totalTokens,
		totalRequests,
		usageData,
		tokenData,
		requestData,
		taskBreakdown,
		providers,
	};
};

export function useDashboardData(filters: DashboardFilters) {
	const [data, setData] = useState<DashboardData | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const fetchData = useCallback(async () => {
		setLoading(true);
		setError(null);

		try {
			// Simulate API call delay
			await new Promise((resolve) => setTimeout(resolve, 800));

			const mockData = generateMockData(filters);
			setData(mockData);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to fetch data");
		} finally {
			setLoading(false);
		}
	}, [
		filters.dateRange?.from,
		filters.dateRange?.to,
		filters.provider,
		filters,
	]);

	useEffect(() => {
		fetchData();
	}, [fetchData]);

	const refresh = useCallback(() => {
		fetchData();
	}, [fetchData]);

	return {
		data,
		loading,
		error,
		refresh,
	};
}
