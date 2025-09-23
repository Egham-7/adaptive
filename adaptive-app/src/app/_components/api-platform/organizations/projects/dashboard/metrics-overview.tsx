"use client";

import { useParams } from "next/navigation";
import {
	FaChartLine,
	FaCoins,
	FaCreditCard,
	FaExclamationTriangle,
	FaServer,
} from "react-icons/fa";
import { api } from "@/trpc/react";
import type { ProjectAnalytics } from "@/types/api-platform/dashboard";
import type { CreditTransactionItem } from "@/types/credits";
import { MetricCardSkeleton } from "./loading-skeleton";
import { VersatileMetricChart } from "./versatile-metric-chart";

interface MetricsOverviewProps {
	data: ProjectAnalytics | null;
	loading: boolean;
}

export function MetricsOverview({ data, loading }: MetricsOverviewProps) {
	const params = useParams();
	const orgId = params.orgId as string;

	// Fetch credit balance and transaction history for credit chart
	const { data: creditBalance } = api.credits.getBalance.useQuery(
		{ organizationId: orgId },
		{ enabled: !!orgId },
	);

	const { data: creditTransactions } =
		api.credits.getTransactionHistory.useQuery(
			{
				organizationId: orgId,
				limit: 30, // Last 30 transactions for chart data
			},
			{ enabled: !!orgId },
		);

	if (loading) {
		return (
			<div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
				{Array.from({ length: 5 }).map((_, i) => (
					<MetricCardSkeleton key={`skeleton-${i}`} />
				))}
			</div>
		);
	}

	if (!data) return null;

	const spendData = data.dailyTrends.map((d) => ({
		date: d.date.toISOString().slice(0, 10),
		value: d.spend, // This is the actual customer spending
	}));

	// Create credit balance history data from transactions
	const creditBalanceData = (creditTransactions?.transactions ?? [])
		.slice() // clone to avoid mutating cached data
		.reverse() // chronological order
		.map((transaction: CreditTransactionItem) => ({
			// Stable date string for charting (YYYY-MM-DD, UTC)
			date: new Date(transaction.createdAt).toISOString().slice(0, 10),
			value: Number.parseFloat(transaction.balanceAfter.toString()),
		}));

	const allMetrics = [
		{
			title: "Credit Balance",
			chartType: "line" as const,
			icon: <FaCreditCard className="h-5 w-5 text-primary" />,
			data: creditBalanceData,
			color: "hsl(var(--primary))",
			totalValue: creditBalance?.formattedBalance || "$0.00",
		},
		{
			title: "Spending Over Time",
			chartType: "line" as const,
			icon: <FaChartLine className="h-5 w-5 text-chart-2" />,
			data: spendData,
			color: "hsl(var(--chart-2))",
			totalValue: `$${(() => {
				const str = data.totalSpend.toString();
				const parts = str.split(".");
				const decimalPart = parts[1] || "";
				const significantDecimals = decimalPart.replace(/0+$/, "").length;
				const decimals = Math.min(Math.max(significantDecimals, 2), 8);
				return data.totalSpend.toFixed(decimals);
			})()}`,
		},
		{
			title: "Token Usage",
			chartType: "bar" as const,
			icon: <FaCoins className="h-5 w-5 text-chart-3" />,
			data: data.dailyTrends.map((d) => ({
				date: d.date.toISOString().slice(0, 10),
				value: d.tokens,
			})),
			color: "hsl(var(--chart-3))",
			totalValue: data.totalTokens.toLocaleString(),
		},
		{
			title: "Request Volume",
			chartType: "area" as const,
			icon: <FaServer className="h-5 w-5 text-chart-4" />,
			data: data.dailyTrends.map((d) => ({
				date: d.date.toISOString().slice(0, 10),
				value: d.requests,
			})),
			color: "hsl(var(--chart-4))",
			totalValue: data.totalRequests.toLocaleString(),
		},
		{
			title: "Error Rate",
			chartType: "area" as const,
			icon: <FaExclamationTriangle className="h-5 w-5 text-destructive" />,
			data: data.dailyTrends.map((d) => ({
				date: d.date.toISOString().slice(0, 10),
				value: d.requests > 0 ? (d.errorCount / d.requests) * 100 : 0,
			})),
			color: "hsl(var(--destructive))",
			totalValue: `${data.errorRate.toFixed(2)}%`,
		},
	];

	return (
		<div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
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
