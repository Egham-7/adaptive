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
	showMarginBreakdown?: boolean;
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
	showMarginBreakdown = false,
}: UsageChartProps) {
	if (!data || data.length === 0) {
		return (
			<div className="flex h-[300px] w-full items-center justify-center text-muted-foreground text-sm">
				No data available
			</div>
		);
	}

	// Calculate margin breakdown if requested
	const chartData = showMarginBreakdown
		? data.map((point) => {
				// Assume provider cost is ~70% of our charge (30% margin)
				const providerCost = point.adaptive * 0.7;
				const ourMargin = point.adaptive * 0.3;
				return {
					...point,
					providerCost,
					ourMargin,
				};
			})
		: data;

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

				{showMarginBreakdown ? (
					<>
						{/* Stacked bars for margin breakdown */}
						<Bar
							dataKey="providerCost"
							stackId="breakdown"
							fill="var(--color-providerCost)"
							radius={[0, 0, 0, 0]}
						/>
						<Bar
							dataKey="ourMargin"
							stackId="breakdown"
							fill="var(--color-ourMargin)"
							radius={[2, 2, 0, 0]}
						/>
						<Bar
							dataKey="singleProvider"
							fill="var(--color-singleProvider)"
							radius={[2, 2, 0, 0]}
							opacity={0.6}
						/>
					</>
				) : (
					<>
						{/* Original bars */}
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
					</>
				)}
			</BarChart>
		</ChartContainer>
	);
}
