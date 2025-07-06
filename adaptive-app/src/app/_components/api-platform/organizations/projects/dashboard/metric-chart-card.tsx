"use client";

import { TrendingDown, TrendingUp } from "lucide-react";
import { Line, LineChart, ResponsiveContainer } from "recharts";

interface MetricChartCardProps {
	title: string;
	value: string;
	change?: string;
	changeType?: "positive" | "negative" | "neutral";
	icon?: React.ReactNode;
	description?: string;
	data: Array<{ value: number }>;
	className?: string;
}

export function MetricChartCard({
	title,
	value,
	change,
	changeType = "neutral",
	icon,
	description,
	data,
	className = "",
}: MetricChartCardProps) {
	const getChangeIcon = () => {
		if (changeType === "positive") return <TrendingUp className="h-4 w-4" />;
		if (changeType === "negative") return <TrendingDown className="h-4 w-4" />;
		return null;
	};

	const getChangeColor = () => {
		if (changeType === "positive") return "text-success";
		if (changeType === "negative") return "text-destructive";
		return "text-muted-foreground";
	};

	const getLineColor = () => {
		if (changeType === "positive") return "#10b981";
		if (changeType === "negative") return "#ef4444";
		return "#6b7280";
	};

	return (
		<div
			className={`rounded-xl border border-gray-200 bg-white p-6 transition-shadow hover:shadow-lg dark:border-[#1F1F23] dark:bg-[#0F0F12] ${className}`}
		>
			<div className="mb-4 flex items-center justify-between">
				{icon && (
					<div className="rounded-lg bg-gray-50 p-2 dark:bg-gray-800">
						{icon}
					</div>
				)}
				{change && (
					<div
						className={`flex items-center gap-1 font-medium text-sm ${getChangeColor()}`}
					>
						{getChangeIcon()}
						{change}
					</div>
				)}
			</div>

			<div className="space-y-3">
				<div>
					<h3 className="font-medium text-gray-600 text-sm dark:text-gray-400">
						{title}
					</h3>
					<p className="font-bold text-2xl text-gray-900 dark:text-white">
						{value}
					</p>
					{description && (
						<p className="text-gray-500 text-xs dark:text-gray-500">
							{description}
						</p>
					)}
				</div>

				<div className="h-16">
					{!data || data.length === 0 ? (
						<div className="flex h-full items-center justify-center text-muted-foreground text-xs">
							No data available
						</div>
					) : (
						<ResponsiveContainer width="100%" height="100%">
							<LineChart data={data}>
								<Line
									type="monotone"
									dataKey="value"
									stroke={getLineColor()}
									strokeWidth={2}
									dot={false}
									activeDot={{ r: 4, fill: getLineColor() }}
								/>
							</LineChart>
						</ResponsiveContainer>
					)}
				</div>
			</div>
		</div>
	);
}
