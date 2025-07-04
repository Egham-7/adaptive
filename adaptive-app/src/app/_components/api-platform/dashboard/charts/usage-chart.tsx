"use client";

import {
	Bar,
	BarChart,
	CartesianGrid,
	Legend,
	ResponsiveContainer,
	XAxis,
	YAxis,
} from "recharts";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { UsageDataPoint } from "@/types/api-platform/dashboard";

interface UsageChartProps {
	data: UsageDataPoint[];
	providerName?: string;
}

export function UsageChart({
	data,
	providerName = "Single Provider",
}: UsageChartProps) {
	return (
		<ChartContainer
			config={{
				adaptive: {
					label: "Adaptive Cost",
					color: "#8b5cf6",
				},
				singleProvider: {
					label: providerName,
					color: "#6b7280",
				},
			}}
			className="h-[300px] w-full"
		>
			<ResponsiveContainer width="100%" height="100%">
				<BarChart
					data={data}
					margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
				>
					<CartesianGrid
						strokeDasharray="3 3"
						className="stroke-gray-200 dark:stroke-gray-700"
					/>
					<XAxis
						dataKey="date"
						className="text-gray-600 dark:text-gray-400"
						fontSize={12}
						axisLine={false}
						tickLine={false}
					/>
					<YAxis
						className="text-gray-600 dark:text-gray-400"
						fontSize={12}
						axisLine={false}
						tickLine={false}
						tickFormatter={(value) => `$${value}`}
					/>
					<ChartTooltip
						content={<ChartTooltipContent />}
						formatter={(value: number | string, name: string) => [
							`$${value}`,
							name === "adaptive" ? "Adaptive" : providerName,
						]}
					/>
					<Legend />
					<Bar
						dataKey="adaptive"
						fill="#8b5cf6"
						radius={[2, 2, 0, 0]}
						name="Adaptive Cost"
					/>
					<Bar
						dataKey="singleProvider"
						fill="#6b7280"
						radius={[2, 2, 0, 0]}
						opacity={0.3}
						name={`${providerName} Cost`}
					/>
				</BarChart>
			</ResponsiveContainer>
		</ChartContainer>
	);
}
