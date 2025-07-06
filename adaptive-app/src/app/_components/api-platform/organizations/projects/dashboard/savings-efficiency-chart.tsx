"use client";

import {
	ResponsiveContainer,
	Scatter,
	ScatterChart,
	XAxis,
	YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { DashboardData } from "@/types/api-platform/dashboard";

interface SavingsEfficiencyChartProps {
	data: DashboardData | null;
	loading: boolean;
}

export function SavingsEfficiencyChart({
	data,
	loading,
}: SavingsEfficiencyChartProps) {
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

	if (!data || !data.taskBreakdown || data.taskBreakdown.length === 0) {
		return (
			<Card>
				<CardHeader>
					<CardTitle>Savings Efficiency by Task</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No savings data available
					</div>
				</CardContent>
			</Card>
		);
	}

	const chartColors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];
	const chartData = data.taskBreakdown.map((task, index) => ({
		name: task.name,
		cost: Number.parseFloat(task.cost.replace("$", "")),
		savingsPercentage: task.percentage,
		requests: Number.parseInt(task.requests.replace(",", "")),
		savings: Number.parseFloat(task.savings.replace("$", "")),
		fill: chartColors[index] || "#6b7280",
	}));

	const maxCost = Math.max(...chartData.map((d) => d.cost));
	const maxSavings = Math.max(...chartData.map((d) => d.savingsPercentage));

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<CardTitle>Savings Efficiency</CardTitle>
					<Badge variant="secondary">Cost vs Savings %</Badge>
				</div>
			</CardHeader>
			<CardContent>
				<div className="mb-6 h-64">
					<ChartContainer
						config={{
							cost: {
								label: "Cost",
								color: "#3b82f6",
							},
							savingsPercentage: {
								label: "Savings %",
								color: "#f59e0b",
							},
						}}
					>
						<ResponsiveContainer width="100%" height="100%">
							<ScatterChart
								margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
								data={chartData}
							>
								<XAxis
									type="number"
									dataKey="cost"
									domain={[0, maxCost * 1.1]}
									axisLine={false}
									tickLine={false}
									className="text-xs"
								/>
								<YAxis
									type="number"
									dataKey="savingsPercentage"
									domain={[0, maxSavings * 1.1]}
									axisLine={false}
									tickLine={false}
									className="text-xs"
								/>
								<ChartTooltip content={<ChartTooltipContent />} />
								<Scatter dataKey="savingsPercentage" fill="#3b82f6" />
							</ScatterChart>
						</ResponsiveContainer>
					</ChartContainer>
				</div>

				<div className="space-y-2">
					<div className="mb-3 text-muted-foreground text-sm">
						Task Performance Summary
					</div>
					{chartData.map((item, _index) => (
						<div
							key={item.name}
							className="flex items-center justify-between rounded-lg bg-muted/50 p-3"
						>
							<div className="flex items-center gap-2">
								<div
									className="h-3 w-3 rounded-full"
									style={{ backgroundColor: item.fill }}
								/>
								<span className="font-medium text-foreground text-sm">
									{item.name}
								</span>
							</div>
							<div className="text-right">
								<Badge variant="outline" className="mb-1">
									{item.savingsPercentage.toFixed(1)}%
								</Badge>
								<div className="text-muted-foreground text-xs">
									${item.cost.toFixed(2)} cost
								</div>
							</div>
						</div>
					))}
				</div>
			</CardContent>
		</Card>
	);
}
