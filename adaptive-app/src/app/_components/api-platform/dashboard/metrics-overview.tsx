"use client";

import { DollarSign, GitBranch, Shield, Zap } from "lucide-react";
import type { DashboardData } from "@/types/api-platform/dashboard";
import { MetricCardSkeleton } from "./loading-skeleton";
import { MetricChartCard } from "./metric-chart-card";

interface MetricsOverviewProps {
	data: DashboardData | null;
	loading: boolean;
}

export function MetricsOverview({ data, loading }: MetricsOverviewProps) {
	if (loading) {
		return (
			<div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
				{Array.from({ length: 4 }).map((_, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: Using index for skeleton components is acceptable
					<MetricCardSkeleton key={`skeleton-${i}`} />
				))}
			</div>
		);
	}

	if (!data) return null;

	const savingsData = data.usageData.map((d) => ({
		value: d.singleProvider - d.adaptive,
	}));

	const spendData = data.usageData.map((d) => ({
		value: d.adaptive,
	}));

	const scalarMetrics = [
		{
			title: "Total Cost Savings",
			value: `$${data.totalSavings.toFixed(2)}`,
			change: `+${data.savingsPercentage.toFixed(1)}%`,
			changeType: "positive" as const,
			icon: <DollarSign className="h-5 w-5 text-green-600" />,
			description: "vs single provider",
			data: savingsData,
		},
		{
			title: "Total Spend",
			value: `$${data.totalSpend.toFixed(2)}`,
			change: "-31.2%",
			changeType: "positive" as const,
			icon: <Zap className="h-5 w-5 text-blue-600" />,
			description: "current period",
			data: spendData,
		},
	];

	const chartMetrics = [
		{
			title: "Total Tokens",
			value: data.totalTokens.toLocaleString(),
			change: "+12.5%",
			changeType: "positive" as const,
			icon: <GitBranch className="h-5 w-5 text-purple-600" />,
			description: "processed",
			data: data.tokenData.map((d) => ({ value: d.tokens })),
		},
		{
			title: "Total Requests",
			value: data.totalRequests.toLocaleString(),
			change: "+8.3%",
			changeType: "positive" as const,
			icon: <Shield className="h-5 w-5 text-orange-600" />,
			description: "completed",
			data: data.requestData.map((d) => ({ value: d.requests })),
		},
	];

	return (
		<div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
			{scalarMetrics.map((metric) => (
				<MetricChartCard key={metric.title} {...metric} />
			))}
			{chartMetrics.map((metric) => (
				<MetricChartCard key={`chart-${metric.title}`} {...metric} />
			))}
		</div>
	);
}
