"use client";

import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import {
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from "@/components/ui/chart";
import type { TokenDataPoint } from "@/types/api-platform/dashboard";

interface TokenChartProps {
	data: TokenDataPoint[];
}

export function TokenChart({ data }: TokenChartProps) {
	return (
		<ChartContainer
			config={{
				tokens: {
					label: "Tokens",
					color: "#f97316",
				},
			}}
			className="h-[60px] w-full"
		>
			<ResponsiveContainer width="100%" height="100%">
				<LineChart data={data}>
					<XAxis dataKey="date" hide />
					<YAxis hide />
					<ChartTooltip
						content={<ChartTooltipContent />}
						formatter={(value: number | string) => [
							Number(value).toLocaleString(),
							"Tokens",
						]}
					/>
					<Line
						type="monotone"
						dataKey="tokens"
						stroke="#f97316"
						strokeWidth={2}
						dot={false}
					/>
				</LineChart>
			</ResponsiveContainer>
		</ChartContainer>
	);
}
