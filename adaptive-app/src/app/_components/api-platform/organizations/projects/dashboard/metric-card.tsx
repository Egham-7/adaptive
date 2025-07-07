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
		if (changeType === "positive") return "text-green-600";
		if (changeType === "negative") return "text-destructive";
		return "text-muted-foreground";
	};

	return (
		<div
			className={`rounded-xl border border-border bg-card p-6 transition-shadow hover:shadow-lg ${className}`}
		>
			<div className="mb-4 flex items-center justify-between">
				{icon && <div className="rounded-lg bg-muted p-2">{icon}</div>}
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
				<h3 className="font-medium text-muted-foreground text-sm">{title}</h3>
				<p className="font-bold text-2xl text-foreground">{value}</p>
				{description && (
					<p className="text-muted-foreground text-xs">{description}</p>
				)}
			</div>
		</div>
	);
}
