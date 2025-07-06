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

	// Calculate trends based on actual data
	const calculateTrend = (data: { date: string; value: number }[]) => {
		if (data.length < 2)
			return { change: null, changeType: "neutral" as const };

		const firstValue = data[0]?.value || 0;
		const lastValue = data[data.length - 1]?.value || 0;

		if (firstValue === 0)
			return { change: null, changeType: "neutral" as const };

		const percentChange = ((lastValue - firstValue) / firstValue) * 100;
		const changeType =
			percentChange > 0
				? ("positive" as const)
				: percentChange < 0
					? ("negative" as const)
					: ("neutral" as const);

		return {
			change: `${percentChange >= 0 ? "+" : ""}${percentChange.toFixed(1)}%`,
			changeType,
		};
	};

	const spendTrend = calculateTrend(spendData);
	const tokenTrend = calculateTrend(
		data.tokenData.map((d) => ({ date: d.date, value: d.tokens })),
	);
	const requestTrend = calculateTrend(
		data.requestData.map((d) => ({ date: d.date, value: d.requests })),
	);

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
			change: spendTrend.change,
			changeType: spendTrend.changeType,
		},
		{
			title: "Token Usage",
			chartType: "bar" as const,
			icon: <FaCoins className="h-5 w-5 text-yellow-700" />,
			data: data.tokenData.map((d) => ({ date: d.date, value: d.tokens })),
			color: "#a16207", // Golden coffee
			totalValue: data.totalTokens.toLocaleString(),
			change: tokenTrend.change,
			changeType: tokenTrend.changeType,
		},
		{
			title: "Request Volume",
			chartType: "area" as const,
			icon: <FaServer className="h-5 w-5 text-stone-700" />,
			data: data.requestData.map((d) => ({ date: d.date, value: d.requests })),
			color: "#78716c", // Espresso brown
			totalValue: data.totalRequests.toLocaleString(),
			change: requestTrend.change,
			changeType: requestTrend.changeType,
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
