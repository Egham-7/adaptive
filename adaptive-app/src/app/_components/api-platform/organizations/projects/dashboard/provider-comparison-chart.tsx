"use client";

import { Bar, BarChart, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { DashboardData } from "@/types/api-platform/dashboard";

interface ProviderComparisonChartProps {
	data: DashboardData | null;
	loading: boolean;
}

const chartConfig = {
	adaptive: {
		label: "Adaptive",
		color: "var(--chart-1)",
	},
	single: {
		label: "Single Provider",
		color: "var(--chart-2)",
	},
} satisfies ChartConfig;

export function ProviderComparisonChart({
	data,
	loading,
}: ProviderComparisonChartProps) {
	if (loading) {
		return (
			<Card>
				<CardContent className="p-6">
					<div className="animate-pulse">
						<div className="mb-4 h-4 w-1/3 rounded bg-muted" />
						<div className="h-48 rounded bg-muted" />
					</div>
				</CardContent>
			</Card>
		);
	}

	if (!data || !data.providers || data.providers.length === 0) {
		return (
			<Card>
				<CardHeader>
					<CardTitle>Provider Comparison</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No provider data available
					</div>
				</CardContent>
			</Card>
		);
	}

	const chartData = data.providers.map((provider, _index) => ({
		name: provider.name.split(" ")[0], // Shortened name for chart display
		fullName: provider.name,
		adaptive: provider.comparisonCosts.adaptive,
		single: provider.comparisonCosts.single,
		savings:
			provider.comparisonCosts.single - provider.comparisonCosts.adaptive,
		savingsPercentage:
			((provider.comparisonCosts.single - provider.comparisonCosts.adaptive) /
				provider.comparisonCosts.single) *
			100,
	}));

	const _maxValue = Math.max(
		...chartData.map((d) => Math.max(d.adaptive, d.single)),
	);

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<CardTitle>Provider Cost Comparison</CardTitle>
					<Badge variant="secondary">Adaptive vs Single</Badge>
				</div>
			</CardHeader>
			<CardContent>
				<div className="mb-6 h-64">
					<ChartContainer config={chartConfig}>
						<BarChart
							accessibilityLayer
							data={chartData}
							margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
						>
							<XAxis
								dataKey="name"
								axisLine={false}
								tickLine={false}
								tickMargin={10}
								fontSize={12}
							/>
							<YAxis
								axisLine={false}
								tickLine={false}
								tickMargin={10}
								fontSize={12}
							/>
							<ChartTooltip content={<ChartTooltipContent />} />
							<Bar
								dataKey="adaptive"
								fill="var(--color-adaptive)"
								radius={[4, 4, 0, 0]}
							/>
							<Bar
								dataKey="single"
								fill="var(--color-single)"
								radius={[4, 4, 0, 0]}
							/>
						</BarChart>
					</ChartContainer>
				</div>

				<div className="space-y-3">
					<div className="flex items-center justify-between text-sm">
						<div className="flex items-center gap-2">
							<div
								className="h-3 w-3 rounded-full"
								style={{ backgroundColor: "var(--color-adaptive)" }}
							/>
							<span className="text-muted-foreground">Adaptive</span>
						</div>
						<div className="flex items-center gap-2">
							<div
								className="h-3 w-3 rounded-full"
								style={{ backgroundColor: "var(--color-single)" }}
							/>
							<span className="text-muted-foreground">Single Provider</span>
						</div>
					</div>

					<div className="grid grid-cols-1 gap-2 text-sm">
						{chartData.map((item, _index) => (
							<div
								key={item.name}
								className="flex items-center justify-between rounded-lg bg-muted/50 p-3"
							>
								<span className="truncate font-medium text-foreground">
									{item.name}
								</span>
								<Badge
									variant="default"
									className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
								>
									{item.savingsPercentage.toFixed(1)}% saved
								</Badge>
							</div>
						))}
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
