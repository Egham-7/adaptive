"use client";

import { useCallback, useState } from "react";
import type { DateRange } from "react-day-picker";

export interface DateRangePreset {
	label: string;
	value: string;
	range: DateRange;
}

function getDaysAgo(days: number): Date {
	return new Date(Date.now() - days * 24 * 60 * 60 * 1000);
}

export function useDateRange(initialRange?: DateRange) {
	const [dateRange, setDateRange] = useState<DateRange>(
		initialRange || {
			from: getDaysAgo(7), // 7 days ago
			to: new Date(),
		},
	);

	const presets: DateRangePreset[] = [
		{
			label: "Last 7 days",
			value: "7d",
			range: {
				from: getDaysAgo(7),
				to: new Date(),
			},
		},
		{
			label: "Last 30 days",
			value: "30d",
			range: {
				from: getDaysAgo(30),
				to: new Date(),
			},
		},
		{
			label: "Last 90 days",
			value: "90d",
			range: {
				from: getDaysAgo(90),
				to: new Date(),
			},
		},
		{
			label: "Last year",
			value: "1y",
			range: {
				from: getDaysAgo(365),
				to: new Date(),
			},
		},
	];

	const setPreset = useCallback(
		(presetValue: string) => {
			const preset = presets.find((p) => p.value === presetValue);
			if (preset) {
				setDateRange(preset.range);
			}
		},
		[],
	);

	const formatDateRange = useCallback((range: DateRange) => {
		if (!range.from || !range.to) return;

		const formatDate = (date: Date) => {
			return date.toLocaleDateString("en-US", {
				month: "short",
				day: "numeric",
				year: "2-digit",
			});
		};
		return `${formatDate(range.from)} - ${formatDate(range.to)}`;
	}, []);

	return {
		dateRange,
		setDateRange,
		presets,
		setPreset,
		formatDateRange,
	};
}
