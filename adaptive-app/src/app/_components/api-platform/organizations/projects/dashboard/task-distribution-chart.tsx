"use client";

import { Cell, Pie, PieChart } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	type ChartConfig,
	ChartContainer,
	ChartLegend,
	ChartLegendContent,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { DashboardData } from "@/types/api-platform/dashboard";

interface TaskDistributionChartProps {
	data: DashboardData | null;
	loading: boolean;
}

const chartConfig = {
	requests: {
		label: "Requests",
	},
	"Code Generation": {
		label: "Code Generation",
		color: "var(--chart-1)",
	},
	"Open Q&A": {
		label: "Open Q&A",
		color: "var(--chart-2)",
	},
	Summarization: {
		label: "Summarization",
		color: "var(--chart-3)",
	},
	Translation: {
		label: "Translation",
		color: "var(--chart-4)",
	},
} satisfies ChartConfig;

export function TaskDistributionChart({
	data,
	loading,
}: TaskDistributionChartProps) {
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
					<div className="flex items-center justify-between">
						<CardTitle>Task Distribution</CardTitle>
						<Badge variant="secondary">0 total</Badge>
					</div>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No task data available
					</div>
				</CardContent>
			</Card>
		);
	}

	const chartColors = [
		"var(--chart-1)",
		"var(--chart-2)",
		"var(--chart-3)",
		"var(--chart-4)",
		"var(--chart-5)",
	];
	const chartData = data.taskBreakdown.map((task, index) => ({
		name: task.name,
		value: Number.parseInt(task.requests.replace(",", "")),
		percentage: task.percentage,
		fill: chartColors[index] || "var(--chart-1)",
	}));

	const totalRequests = chartData.reduce((sum, item) => sum + item.value, 0);

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<CardTitle>Task Distribution</CardTitle>
					<Badge variant="secondary">
						{totalRequests.toLocaleString()} total
					</Badge>
				</div>
			</CardHeader>
			<CardContent>
				<ChartContainer config={chartConfig}>
					<PieChart accessibilityLayer>
						<Pie
							data={chartData}
							cx="50%"
							cy="50%"
							labelLine={false}
							outerRadius={80}
							dataKey="value"
						>
							{chartData.map((entry) => (
								<Cell key={entry.name} fill={entry.fill} />
							))}
						</Pie>
						<ChartTooltip content={<ChartTooltipContent />} />
						<ChartLegend
							content={<ChartLegendContent />}
							verticalAlign="bottom"
							className="pt-4"
						/>
					</PieChart>
				</ChartContainer>
			</CardContent>
		</Card>
	);
}
