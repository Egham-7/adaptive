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
					color: "#b45309", // Coffee brown
				},
				singleProvider: {
					label: `${providerName} Cost`,
					color: "#78716c", // Espresso brown (lighter for comparison)
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
					<ChartTooltip content={<ChartTooltipContent />} />
					<Legend />
					<Bar
						dataKey="adaptive"
						fill="#b45309"
						radius={[2, 2, 0, 0]}
						name="Adaptive Cost"
					/>
					<Bar
						dataKey="singleProvider"
						fill="#78716c"
						radius={[2, 2, 0, 0]}
						opacity={0.6}
						name={`${providerName} Cost`}
					/>
				</BarChart>
			</ResponsiveContainer>
		</ChartContainer>
	);
}
