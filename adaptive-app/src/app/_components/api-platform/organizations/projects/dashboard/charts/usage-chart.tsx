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
import type { DailyTrendDataPoint } from "@/types/api-platform/dashboard";

interface UsageChartProps {
	data: DailyTrendDataPoint[];
	selectedModels?: string[];
	providerName?: string;
}

// Color palette for different models
const getColorForModel = (index: number): string => {
	const colors = [
		"var(--chart-2)",
		"var(--chart-3)",
		"var(--chart-4)",
		"var(--chart-5)",
		"#8884d8",
		"#82ca9d",
		"#ffc658",
		"#ff7300",
		"#00ff00",
		"#ff00ff",
	];
	return colors[index % colors.length] || "var(--chart-2)";
};

// Helper function to format model names for display
const formatModelName = (modelId: string) => {
	return modelId
		.split("-")
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(" ");
};

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
	selectedModels = [],
	providerName = "Direct Cost",
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

	// Create dynamic chart config for selected models
	const dynamicConfig: Record<string, { label: string; color: string }> = {
		...chartConfig,
		singleProvider: {
			...chartConfig.singleProvider,
			label: `${providerName} Cost`,
		},
	};

	// Add config entries for each selected model
	selectedModels.forEach((modelId, index) => {
		dynamicConfig[modelId] = {
			label: formatModelName(modelId),
			color: getColorForModel(index),
		};
	});

	return (
		<ChartContainer config={dynamicConfig} className="h-[300px] w-full">
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

				{/* Always show Adaptive cost bar */}
				<Bar
					dataKey="adaptive"
					fill="var(--color-adaptive)"
					radius={[2, 2, 0, 0]}
				/>

				{/* Dynamically render bars for selected models */}
				{selectedModels.map((modelId, index) => (
					<Bar
						key={modelId}
						dataKey={modelId}
						fill={getColorForModel(index)}
						radius={[2, 2, 0, 0]}
						opacity={0.7}
					/>
				))}

				{/* Legacy single provider bar - only show if no models selected */}
				{selectedModels.length === 0 && (
					<Bar
						dataKey="singleProvider"
						fill="var(--color-singleProvider)"
						radius={[2, 2, 0, 0]}
						opacity={0.6}
					/>
				)}
			</BarChart>
		</ChartContainer>
	);
}
