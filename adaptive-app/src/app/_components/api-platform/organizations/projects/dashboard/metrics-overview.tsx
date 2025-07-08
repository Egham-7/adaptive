"use client";

import {
	FaChartLine,
	FaCoins,
	FaDollarSign,
	FaExclamationTriangle,
	FaServer,
} from "react-icons/fa";
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
			<div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
				{Array.from({ length: 5 }).map((_, i) => (
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
			icon: <FaDollarSign className="h-5 w-5 text-success" />,
			data: savingsData,
			color: "hsl(var(--chart-1))",
			totalValue: `$${data.totalSavings.toFixed(2)}`,
		},
		{
			title: "Spending Over Time",
			chartType: "line" as const,
			icon: <FaChartLine className="h-5 w-5 text-chart-2" />,
			data: spendData,
			color: "hsl(var(--chart-2))",
			totalValue: `$${data.totalSpend.toFixed(2)}`,
		},
		{
			title: "Token Usage",
			chartType: "bar" as const,
			icon: <FaCoins className="h-5 w-5 text-chart-3" />,
			data: data.tokenData.map((d) => ({ date: d.date, value: d.tokens })),
			color: "hsl(var(--chart-3))",
			totalValue: data.totalTokens.toLocaleString(),
		},
		{
			title: "Request Volume",
			chartType: "area" as const,
			icon: <FaServer className="h-5 w-5 text-chart-4" />,
			data: data.requestData.map((d) => ({ date: d.date, value: d.requests })),
			color: "hsl(var(--chart-4))",
			totalValue: data.totalRequests.toLocaleString(),
		},
		{
			title: "Error Rate",
			chartType: "area" as const,
			icon: <FaExclamationTriangle className="h-5 w-5 text-destructive" />,
			data: data.errorRateData.map((d) => ({
				date: d.date,
				value: d.errorRate,
			})),
			color: "hsl(var(--destructive))",
			totalValue: `${data.errorRate.toFixed(2)}%`,
		},
	];

	return (
		<div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
			{allMetrics.map((metric) => (
				<VersatileMetricChart
					key={metric.title}
					title={metric.title}
					chartType={metric.chartType}
					data={metric.data}
					icon={metric.icon}
					color={metric.color}
					totalValue={metric.totalValue}
				/>
			))}
		</div>
	);
}
