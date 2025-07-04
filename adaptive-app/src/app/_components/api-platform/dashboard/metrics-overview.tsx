"use client";

import { FaChartLine, FaCoins, FaDollarSign, FaServer } from "react-icons/fa";
import type { DashboardData } from "@/types/api-platform/dashboard";
import { MetricCardSkeleton } from "./loading-skeleton";
import { VersatileMetricChart } from "./versatile-metric-chart";

interface MetricsOverviewProps {
	data: DashboardData | null;
	loading: boolean;
}

export function MetricsOverview({ data, loading }: MetricsOverviewProps) {
	if (loading) {
		return (
			<div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
				{Array.from({ length: 4 }).map((_, i) => (
					// biome-ignore lint/suspicious/noArrayIndexKey: Using index for skeleton components is acceptable
					<MetricCardSkeleton key={`skeleton-${i}`} />
				))}
			</div>
		);
	}

	if (!data) return null;

	const savingsData = data.usageData.map((d) => ({
		date: d.date,
		value: d.singleProvider - d.adaptive,
	}));

	const spendData = data.usageData.map((d) => ({
		date: d.date,
		value: d.adaptive,
	}));

	const allMetrics = [
		{
			title: "Cost Savings Trend",
			chartType: "area" as const,
			icon: <FaDollarSign className="h-5 w-5 text-amber-700" />,
			data: savingsData,
			color: "#b45309", // Coffee brown
			totalValue: `$${data.totalSavings.toFixed(2)}`,
			change: `+${data.savingsPercentage.toFixed(1)}%`,
			changeType: "positive" as const,
		},
		{
			title: "Spending Over Time",
			chartType: "line" as const,
			icon: <FaChartLine className="h-5 w-5 text-orange-700" />,
			data: spendData,
			color: "#c2410c", // Burnt orange
			totalValue: `$${data.totalSpend.toFixed(2)}`,
			change: "-31.2%",
			changeType: "positive" as const,
		},
		{
			title: "Token Usage",
			chartType: "bar" as const,
			icon: <FaCoins className="h-5 w-5 text-yellow-700" />,
			data: data.tokenData.map((d) => ({ date: d.date, value: d.tokens })),
			color: "#a16207", // Golden coffee
			totalValue: data.totalTokens.toLocaleString(),
			change: "+12.5%",
			changeType: "positive" as const,
		},
		{
			title: "Request Volume",
			chartType: "area" as const,
			icon: <FaServer className="h-5 w-5 text-stone-700" />,
			data: data.requestData.map((d) => ({ date: d.date, value: d.requests })),
			color: "#78716c", // Espresso brown
			totalValue: data.totalRequests.toLocaleString(),
			change: "+8.3%",
			changeType: "positive" as const,
		},
	];

	return (
		<div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
			{allMetrics.map((metric) => (
				<VersatileMetricChart
					key={metric.title}
					title={metric.title}
					chartType={metric.chartType}
					data={metric.data}
					icon={metric.icon}
					color={metric.color}
					totalValue={metric.totalValue}
					change={metric.change}
					changeType={metric.changeType}
				/>
			))}
		</div>
	);
}
