"use client";

import { TrendingDown, TrendingUp } from "lucide-react";
import {
	Area,
	AreaChart,
	Bar,
	BarChart,
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	XAxis,
	YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";

interface MetricChartProps {
	title: string;
	chartType: "line" | "area" | "bar";
	data: Array<{ date: string; value: number }>;
	icon: React.ReactNode;
	color: string;
	totalValue: string;
	change: string;
	changeType: "positive" | "negative" | "neutral";
}

export function VersatileMetricChart({
	title,
	chartType,
	data,
	icon,
	color,
	totalValue,
	change,
	changeType,
}: MetricChartProps) {
	const chartConfig = {
		value: {
			label: "Value",
			color: color,
		},
	};

	const getChangeIcon = () => {
		if (changeType === "positive") return <TrendingUp className="h-4 w-4" />;
		if (changeType === "negative") return <TrendingDown className="h-4 w-4" />;
		return null;
	};

	const getChangeColor = () => {
		if (changeType === "positive") return "text-success";
		if (changeType === "negative") return "text-destructive";
		return "text-muted-foreground";
	};

	const renderChart = () => {
		const commonProps = {
			data,
			margin: { top: 5, right: 5, left: 5, bottom: 5 },
		};

		switch (chartType) {
			case "area":
				return (
					<AreaChart {...commonProps}>
						<CartesianGrid
							strokeDasharray="3 3"
							className="stroke-gray-200 dark:stroke-gray-700"
						/>
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
						/>
						<YAxis fontSize={10} axisLine={false} tickLine={false} />
						<ChartTooltip content={<ChartTooltipContent />} />
						<Area
							type="monotone"
							dataKey="value"
							fill={color}
							fillOpacity={0.3}
							stroke={color}
							strokeWidth={2}
							name="Value"
						/>
					</AreaChart>
				);
			case "bar":
				return (
					<BarChart {...commonProps}>
						<CartesianGrid
							strokeDasharray="3 3"
							className="stroke-gray-200 dark:stroke-gray-700"
						/>
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
						/>
						<YAxis fontSize={10} axisLine={false} tickLine={false} />
						<ChartTooltip content={<ChartTooltipContent />} />
						<Bar
							dataKey="value"
							fill={color}
							radius={[2, 2, 0, 0]}
							name="Value"
						/>
					</BarChart>
				);
			default:
				return (
					<LineChart {...commonProps}>
						<CartesianGrid
							strokeDasharray="3 3"
							className="stroke-gray-200 dark:stroke-gray-700"
						/>
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
						/>
						<YAxis fontSize={10} axisLine={false} tickLine={false} />
						<ChartTooltip content={<ChartTooltipContent />} />
						<Line
							type="monotone"
							dataKey="value"
							stroke={color}
							strokeWidth={2}
							dot={false}
							name="Value"
						/>
					</LineChart>
				);
		}
	};

	return (
		<Card>
			<CardHeader className="pb-2">
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-2">
						<div className="rounded-lg bg-gray-50 p-2 dark:bg-gray-800">
							{icon}
						</div>
						<CardTitle className="font-medium text-sm">{title}</CardTitle>
					</div>
					<div
						className={`flex items-center gap-1 font-medium text-sm ${getChangeColor()}`}
					>
						{getChangeIcon()}
						{change}
					</div>
				</div>
				<div className="mt-2">
					<div className="font-bold text-2xl text-gray-900 dark:text-white">
						{totalValue}
					</div>
				</div>
			</CardHeader>
			<CardContent>
				<ChartContainer config={chartConfig} className="h-[120px] w-full">
					{!data || data.length === 0 ? (
						<div className="flex h-full items-center justify-center text-muted-foreground text-sm">
							No data available
						</div>
					) : (
						<ResponsiveContainer width="100%" height="100%">
							{renderChart()}
						</ResponsiveContainer>
					)}
				</ChartContainer>
			</CardContent>
		</Card>
	);
}
