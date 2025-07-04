"use client";

import { TrendingDown, TrendingUp } from "lucide-react";
import type { MetricCardData } from "@/types/api-platform/dashboard";

interface MetricCardProps extends MetricCardData {
	className?: string;
}

export function MetricCard({
	title,
	value,
	change,
	changeType = "neutral",
	icon,
	description,
	className = "",
}: MetricCardProps) {
	const getChangeIcon = () => {
		if (changeType === "positive") return <TrendingUp className="h-4 w-4" />;
		if (changeType === "negative") return <TrendingDown className="h-4 w-4" />;
		return null;
	};

	const getChangeColor = () => {
		if (changeType === "positive") return "text-green-600 dark:text-green-400";
		if (changeType === "negative") return "text-red-600 dark:text-red-400";
		return "text-gray-600 dark:text-gray-400";
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

			<div className="space-y-1">
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
		</div>
	);
}
