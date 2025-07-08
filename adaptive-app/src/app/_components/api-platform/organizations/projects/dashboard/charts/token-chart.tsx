"use client";

import { Line, LineChart, XAxis, YAxis } from "recharts";
import {
	type ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { TokenDataPoint } from "@/types/api-platform/dashboard";

const chartConfig = {
	tokens: {
		label: "Tokens",
		color: "var(--chart-2)",
	},
} satisfies ChartConfig;

interface TokenChartProps {
	data: TokenDataPoint[];
}

export function TokenChart({ data }: TokenChartProps) {
	if (!data || data.length === 0) {
		return (
			<div className="flex h-[60px] w-full items-center justify-center text-muted-foreground text-sm">
				No data available
			</div>
		);
	}

	return (
		<ChartContainer config={chartConfig} className="h-[60px] w-full">
			<LineChart accessibilityLayer data={data}>
				<XAxis dataKey="date" hide />
				<YAxis hide />
				<ChartTooltip content={<ChartTooltipContent />} />
				<Line
					type="monotone"
					dataKey="tokens"
					stroke="var(--color-tokens)"
					strokeWidth={2}
					dot={false}
				/>
			</LineChart>
		</ChartContainer>
	);
}
