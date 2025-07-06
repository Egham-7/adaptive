"use client";

import { Cell, Legend, Pie, PieChart, ResponsiveContainer } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { DashboardData } from "@/types/api-platform/dashboard";

interface TaskDistributionChartProps {
	data: DashboardData | null;
	loading: boolean;
}

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

	const chartColors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];
	const chartData = data.taskBreakdown.map((task, index) => ({
		name: task.name,
		value: Number.parseInt(task.requests.replace(",", "")),
		percentage: task.percentage,
		fill: chartColors[index] || "#6b7280",
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
				<ChartContainer
					config={{
						requests: {
							label: "Requests",
						},
						"Code Generation": {
							label: "Code Generation",
							color: "#3b82f6",
						},
						"Open Q&A": {
							label: "Open Q&A",
							color: "#f59e0b",
						},
						Summarization: {
							label: "Summarization",
							color: "#10b981",
						},
						Translation: {
							label: "Translation",
							color: "#ef4444",
						},
					}}
				>
					<ResponsiveContainer width="100%" height="100%">
						<PieChart>
							<Pie
								data={chartData}
								cx="50%"
								cy="50%"
								labelLine={false}
								outerRadius={80}
								fill="#8884d8"
								dataKey="value"
							>
								{chartData.map((entry) => (
									<Cell key={entry.name} fill={entry.fill} />
								))}
							</Pie>
							<ChartTooltip content={<ChartTooltipContent />} />
							<Legend
								verticalAlign="bottom"
								height={36}
								iconType="circle"
								wrapperStyle={{ paddingTop: "20px" }}
							/>
						</PieChart>
					</ResponsiveContainer>
				</ChartContainer>
			</CardContent>
		</Card>
	);
}
