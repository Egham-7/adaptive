"use client";

import { Bar, BarChart, XAxis, YAxis } from "recharts";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { RequestDataPoint } from "@/types/api-platform/dashboard";

const chartConfig = {
	requests: {
		label: "Requests",
		color: "var(--chart-1)",
	},
} satisfies ChartConfig;

interface RequestsChartProps {
	data: RequestDataPoint[];
}

export function RequestsChart({ data }: RequestsChartProps) {
	if (!data || data.length === 0) {
		return (
			<div className="flex h-[60px] w-full items-center justify-center text-muted-foreground text-sm">
				No data available
			</div>
		);
	}

	return (
		<ChartContainer config={chartConfig} className="h-[60px] w-full">
			<BarChart accessibilityLayer data={data}>
				<XAxis dataKey="date" hide />
				<YAxis hide />
				<ChartTooltip content={<ChartTooltipContent />} />
				<Bar
					dataKey="requests"
					fill="var(--color-requests)"
					radius={[1, 1, 0, 0]}
				/>
			</BarChart>
		</ChartContainer>
	);
}
