"use client";

import {
	Area,
	AreaChart,
	Bar,
	BarChart,
	CartesianGrid,
	Line,
	LineChart,
	XAxis,
	YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	type ChartConfig,
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
}

export function VersatileMetricChart({
	title,
	chartType,
	data,
	icon,
	color,
	totalValue,
}: MetricChartProps) {
	const chartConfig = {
		value: {
			label: "Value",
			color: color,
		},
	} satisfies ChartConfig;

	const renderChart = () => {
		const commonProps = {
			data,
			margin: { top: 5, right: 5, left: 5, bottom: 5 },
		};

		switch (chartType) {
			case "area":
				return (
					<AreaChart accessibilityLayer {...commonProps}>
						<CartesianGrid vertical={false} />
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<YAxis
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<ChartTooltip content={<ChartTooltipContent />} />
						<Area
							type="monotone"
							dataKey="value"
							fill="var(--color-value)"
							fillOpacity={0.3}
							stroke="var(--color-value)"
							strokeWidth={2}
						/>
					</AreaChart>
				);
			case "bar":
				return (
					<BarChart accessibilityLayer {...commonProps}>
						<CartesianGrid vertical={false} />
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<YAxis
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<ChartTooltip content={<ChartTooltipContent />} />
						<Bar
							dataKey="value"
							fill="var(--color-value)"
							radius={[2, 2, 0, 0]}
						/>
					</BarChart>
				);
			default:
				return (
					<LineChart accessibilityLayer {...commonProps}>
						<CartesianGrid vertical={false} />
						<XAxis
							dataKey="date"
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<YAxis
							fontSize={10}
							axisLine={false}
							tickLine={false}
							tickMargin={10}
						/>
						<ChartTooltip content={<ChartTooltipContent />} />
						<Line
							type="monotone"
							dataKey="value"
							stroke="var(--color-value)"
							strokeWidth={2}
							dot={false}
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
				</div>
				<div className="mt-2">
					<div className="font-bold text-2xl text-gray-900 dark:text-white">
						{totalValue}
					</div>
				</div>
			</CardHeader>
			<CardContent>
				<ChartContainer config={chartConfig} className="h-[140px] w-full">
					{!data || data.length === 0 ? (
						<div className="flex h-full items-center justify-center text-muted-foreground text-sm">
							No data available
						</div>
					) : (
						renderChart()
					)}
				</ChartContainer>
			</CardContent>
		</Card>
	);
}
