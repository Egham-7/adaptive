"use client";

import { useParams } from "next/navigation";
import {
	FaChartLine,
	FaCoins,
	FaCreditCard,
	FaDollarSign,
	FaExclamationTriangle,
	FaServer,
} from "react-icons/fa";
import { api } from "@/trpc/react";
import type { DashboardData } from "@/types/api-platform/dashboard";
import { formatCurrencyWithDynamicPrecision } from "@/utils/formatting";
import { MetricCardSkeleton } from "./loading-skeleton";
import { VersatileMetricChart } from "./versatile-metric-chart";

// Calculate direct cost for a specific model using actual token usage
function calculateDirectModelCost(
	usageData: { inputTokens: number; outputTokens: number }[],
	modelId: string,
	pricingData:
		| Record<string, { inputCost: number; outputCost: number }>
		| undefined,
): number {
	if (!pricingData || !pricingData[modelId]) {
		console.warn(`No pricing data found for model: ${modelId}`);
		return 0;
	}

	const modelPricing = pricingData[modelId];

	return usageData.reduce((totalCost, usage) => {
		const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputCost;
		const outputCost =
			(usage.outputTokens / 1_000_000) * modelPricing.outputCost;
		return totalCost + inputCost + outputCost;
	}, 0);
}

interface MetricsOverviewProps {
	data: DashboardData | null;
	loading: boolean;
	selectedModel?: string;
}

export function MetricsOverview({
	data,
	loading,
	selectedModel = "gpt-4o",
}: MetricsOverviewProps) {
	const params = useParams();
	const orgId = params.orgId as string;

	// Fetch dynamic pricing data
	const { data: modelPricing, isLoading: pricingLoading } =
		api.modelPricing.getAllModelPricing.useQuery();

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

	if (loading || pricingLoading) {
		return (
			<div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
				{Array.from({ length: 6 }).map((_, i) => (
					<MetricCardSkeleton key={`skeleton-${i}`} />
				))}
			</div>
		);
	}

	if (!data) return null;

	// Calculate dynamic costs for the selected model
	const usageDataWithDynamicCosts = data.usageData.map((d) => {
		const adaptiveCost = d.adaptive;
		const modelCost = calculateDirectModelCost(
			[{ inputTokens: d.inputTokens || 0, outputTokens: d.outputTokens || 0 }],
			selectedModel,
			modelPricing,
		);
		return {
			...d,
			adaptiveCost,
			modelCost,
			savings: Math.max(0, modelCost - adaptiveCost),
		};
	});

	const savingsData = usageDataWithDynamicCosts.map((d) => ({
		date: d.date,
		value: d.savings,
	}));

	const spendData = data.usageData.map((d) => ({
		date: d.date,
		value: d.adaptive, // This is the actual customer spending (same as adaptive line in usage chart)
	}));

	// Create credit balance history data from transactions
	const creditBalanceData = creditTransactions?.transactions
		? creditTransactions.transactions
				.reverse() // Reverse to show chronological order
				.map((transaction) => ({
					date: new Date(transaction.createdAt).toLocaleDateString(),
					value: Number.parseFloat(transaction.balanceAfter.toString()),
				}))
		: [];

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
			title: "Cost Savings Trend",
			chartType: "area" as const,
			icon: <FaDollarSign className="h-5 w-5 text-success" />,
			data: savingsData,
			color: "hsl(var(--chart-1))",
			totalValue: formatCurrencyWithDynamicPrecision(
				usageDataWithDynamicCosts.reduce((sum, d) => sum + d.savings, 0),
			),
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
