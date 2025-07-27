"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PieChart } from "lucide-react";
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
	const [showMarginBreakdown, setShowMarginBreakdown] = useState(false);

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
					<div className="flex-1">
						<div className="flex items-center justify-between mb-1">
							<CardTitle>Total Spend</CardTitle>
							<div className="flex items-center gap-2">
								{showMarginBreakdown && (
									<Badge variant="secondary" className="text-xs">
										Margin View
									</Badge>
								)}
								<Button
									variant="outline"
									size="sm"
									onClick={() => setShowMarginBreakdown(!showMarginBreakdown)}
									className="h-8 px-3"
								>
									<PieChart className="h-3 w-3 mr-1" />
									{showMarginBreakdown ? "Cost View" : "Margin View"}
								</Button>
							</div>
						</div>
						<div className="flex items-baseline gap-4">
							<span className="font-bold text-3xl text-foreground">
								$
								{typeof totalSpend === "number"
									? totalSpend < 0.01 && totalSpend > 0
										? totalSpend.toFixed(6)
										: totalSpend.toFixed(2)
									: totalSpend}
							</span>
							<span className="text-muted-foreground text-sm">
								vs $
								{typeof totalComparison === "number"
									? totalComparison < 0.01 && totalComparison > 0
										? totalComparison.toFixed(6)
										: totalComparison.toFixed(2)
									: totalComparison}{" "}
								({currentProvider?.name})
							</span>
						</div>
						<div className="mt-2 flex items-center gap-2">
							<span className="font-medium text-sm text-success">
								$
								{typeof totalSavings === "number"
									? totalSavings < 0.01 && totalSavings > 0
										? totalSavings.toFixed(6)
										: totalSavings.toFixed(2)
									: totalSavings}{" "}
								saved ({savingsPercentage}%)
							</span>
						</div>

						{showMarginBreakdown && (
							<div className="mt-3 p-3 bg-muted/30 rounded-lg">
								<h4 className="font-medium text-sm mb-2">Revenue Breakdown (Estimated)</h4>
								<div className="grid grid-cols-2 gap-4 text-sm">
									<div>
										<div className="text-muted-foreground">Provider Cost (~70%)</div>
										<div className="font-mono font-medium">
											${(totalSpend * 0.7).toFixed(4)}
										</div>
									</div>
									<div>
										<div className="text-muted-foreground">Our Margin (~30%)</div>
										<div className="font-mono font-medium text-green-600">
											${(totalSpend * 0.3).toFixed(4)}
										</div>
									</div>
								</div>
							</div>
						)}
					</div>
				</div>
			</CardHeader>
			<CardContent>
				<UsageChart
					data={data.usageData}
					providerName={currentProvider?.name}
					showMarginBreakdown={showMarginBreakdown}
				/>
			</CardContent>
		</Card>
	);
}
