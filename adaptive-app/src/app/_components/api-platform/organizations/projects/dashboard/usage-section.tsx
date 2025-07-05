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

	if (!data) return null;

	const currentProvider = providers.find((p) => p.id === selectedProvider);
	const totalSpend = data.totalSpend;
	const totalComparison = currentProvider?.comparisonCosts.single || 0;
	const totalSavings = totalComparison - totalSpend;
	const savingsPercentage = ((totalSavings / totalComparison) * 100).toFixed(1);

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<div>
						<CardTitle className="mb-1">Total Spend</CardTitle>
						<div className="flex items-baseline gap-4">
							<span className="font-bold text-3xl text-foreground">
								${totalSpend.toFixed(2)}
							</span>
							<span className="text-muted-foreground text-sm">
								vs ${totalComparison.toFixed(2)} ({currentProvider?.name})
							</span>
						</div>
						<div className="mt-2 flex items-center gap-2">
							<span className="font-medium text-green-600 text-sm dark:text-green-400">
								${totalSavings.toFixed(2)} saved ({savingsPercentage}%)
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
