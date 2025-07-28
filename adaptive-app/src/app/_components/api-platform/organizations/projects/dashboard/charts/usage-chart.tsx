"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
	type ChartConfig,
	ChartContainer,
	ChartLegend,
	ChartLegendContent,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { UsageDataPoint } from "@/types/api-platform/dashboard";

interface UsageChartProps {
	data: UsageDataPoint[];
	providerName?: string;
}

const chartConfig = {
	adaptive: {
		label: "Adaptive Cost",
		color: "var(--chart-1)",
	},
	singleProvider: {
		label: "Single Provider Cost",
		color: "var(--chart-2)",
	},
	providerCost: {
		label: "Provider Cost",
		color: "var(--chart-3)",
	},
	ourMargin: {
		label: "Our Margin",
		color: "var(--chart-4)",
	},
} satisfies ChartConfig;

export function UsageChart({
	data,
	providerName = "Single Provider",
}: UsageChartProps) {
	if (!data || data.length === 0) {
		return (
			<div className="flex h-[300px] w-full items-center justify-center text-muted-foreground text-sm">
				No data available
			</div>
		);
	}

	// Use data as-is for cost comparison chart
	const chartData = data;

	const config = {
		...chartConfig,
		singleProvider: {
			...chartConfig.singleProvider,
			label: `${providerName} Cost`,
		},
	};

	return (
		<ChartContainer config={config} className="h-[300px] w-full">
			<BarChart
				accessibilityLayer
				data={chartData}
				margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
			>
				<CartesianGrid vertical={false} />
				<XAxis
					dataKey="date"
					tickLine={false}
					axisLine={false}
					tickMargin={10}
					fontSize={12}
				/>
				<YAxis
					tickLine={false}
					axisLine={false}
					tickMargin={10}
					fontSize={12}
					tickFormatter={(value) => `$${value.toFixed(2)}`}
				/>
				<ChartTooltip
					content={
						<ChartTooltipContent
							formatter={(value, name) => [
								`$${Number(value).toFixed(4)}`,
								name,
							]}
						/>
					}
				/>
				<ChartLegend content={<ChartLegendContent />} />

				{/* Cost comparison bars */}
				<Bar
					dataKey="adaptive"
					fill="var(--color-adaptive)"
					radius={[2, 2, 0, 0]}
				/>
				<Bar
					dataKey="singleProvider"
					fill="var(--color-singleProvider)"
					radius={[2, 2, 0, 0]}
					opacity={0.6}
				/>
			</BarChart>
		</ChartContainer>
	);
}
