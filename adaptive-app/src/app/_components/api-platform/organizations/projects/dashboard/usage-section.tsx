"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";
import { UsageChart } from "./charts/usage-chart";
import { ChartSkeleton } from "./loading-skeleton";

interface UsageSectionProps {
	data: DashboardData | null;
	loading: boolean;
	selectedProvider: string;
	providers: Provider[];
}

export function UsageSection({
	data,
	loading,
	selectedProvider,
	providers,
}: UsageSectionProps) {
	if (loading) {
		return <ChartSkeleton />;
	}

	if (!data || !data.usageData || data.usageData.length === 0) {
		return (
			<Card>
				<CardHeader>
					<CardTitle>Usage vs Single Provider Cost</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No usage data available
					</div>
				</CardContent>
			</Card>
		);
	}

	const currentProvider = providers.find((p) => p.id === selectedProvider);
	const totalSpend = data.totalSpend;
	const totalComparison = currentProvider?.comparisonCosts.single || 0;
	const totalSavings = totalComparison - totalSpend;
	const savingsPercentage =
		totalComparison > 0
			? ((totalSavings / totalComparison) * 100).toFixed(1)
			: "0.0";

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<div>
						<CardTitle className="mb-1">Total Spend</CardTitle>
						<div className="flex items-baseline gap-4">
							<span className="font-bold text-3xl text-foreground">
							${typeof totalSpend === 'number' ? (totalSpend < 0.01 && totalSpend > 0 ? totalSpend.toFixed(6) : totalSpend.toFixed(2)) : totalSpend}
							</span>
							<span className="text-muted-foreground text-sm">
							vs ${typeof totalComparison === 'number' ? (totalComparison < 0.01 && totalComparison > 0 ? totalComparison.toFixed(6) : totalComparison.toFixed(2)) : totalComparison} ({currentProvider?.name})
							</span>
						</div>
						<div className="mt-2 flex items-center gap-2">
							<span className="font-medium text-sm text-success">
							${typeof totalSavings === 'number' ? (totalSavings < 0.01 && totalSavings > 0 ? totalSavings.toFixed(6) : totalSavings.toFixed(2)) : totalSavings} saved ({savingsPercentage}%)
							</span>
						</div>
					</div>
				</div>
			</CardHeader>
			<CardContent>
				<UsageChart
					data={data.usageData}
					providerName={currentProvider?.name}
				/>
			</CardContent>
		</Card>
	);
}
