"use client";

import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { RequestDataPoint } from "@/types/api-platform/dashboard";

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
		<ChartContainer
			config={{
				requests: {
					label: "Requests",
					color: "#8b5cf6",
				},
			}}
			className="h-[60px] w-full"
		>
			<ResponsiveContainer width="100%" height="100%">
				<BarChart data={data}>
					<XAxis dataKey="date" hide />
					<YAxis hide />
					<ChartTooltip content={<ChartTooltipContent />} />
					<Bar dataKey="requests" fill="#8b5cf6" radius={[1, 1, 0, 0]} />
				</BarChart>
			</ResponsiveContainer>
		</ChartContainer>
	);
}
