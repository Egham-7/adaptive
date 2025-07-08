"use client";

import { Area, AreaChart, XAxis, YAxis } from "recharts";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { ErrorRateDataPoint } from "@/types/api-platform/dashboard";

const chartConfig = {
	errorRate: {
		label: "Error Rate (%)",
		color: "hsl(var(--destructive))",
	},
} satisfies ChartConfig;

interface ErrorRateChartProps {
	data: ErrorRateDataPoint[];
}

export function ErrorRateChart({ data }: ErrorRateChartProps) {
	if (!data || data.length === 0) {
		return (
			<div className="flex h-[60px] w-full items-center justify-center text-muted-foreground text-sm">
				No data available
			</div>
		);
	}

	return (
		<ChartContainer config={chartConfig} className="h-[60px] w-full">
			<AreaChart accessibilityLayer data={data}>
				<XAxis dataKey="date" hide />
				<YAxis hide />
				<ChartTooltip content={<ChartTooltipContent />} />
				<Area
					dataKey="errorRate"
					type="monotone"
					fill="var(--color-errorRate)"
					fillOpacity={0.4}
					stroke="var(--color-errorRate)"
					strokeWidth={2}
				/>
			</AreaChart>
		</ChartContainer>
	);
}
