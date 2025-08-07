import { Download } from "lucide-react";
import type { DateRange } from "react-day-picker";
import { Button } from "@/components/ui/button";
import { DateRangePicker } from "@/components/ui/date-range-picker";

interface DashboardHeaderProps {
	dateRange: DateRange;
	onDateRangeChange: (range: DateRange) => void;
	onExport: () => void;
}

export function DashboardHeader({
	dateRange,
	onDateRangeChange,
	onExport,
}: DashboardHeaderProps) {
	return (
		<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
			<div className="flex-1">
				<h1 className="font-bold text-2xl text-foreground sm:text-3xl">
					Usage Dashboard
				</h1>
				<p className="mt-1 text-muted-foreground text-sm sm:text-base">
					Monitor your API usage and optimize costs
				</p>
			</div>
			<div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center sm:gap-3">
				<DateRangePicker
					dateRange={dateRange}
					onDateRangeChange={onDateRangeChange}
					presetRanges={[
						{
							label: "Last 7 days",
							value: "7d",
							range: {
								from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
								to: new Date(),
							},
						},
						{
							label: "Last 30 days",
							value: "30d",
							range: {
								from: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
								to: new Date(),
							},
						},
						{
							label: "Last 90 days",
							value: "90d",
							range: {
								from: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
								to: new Date(),
							},
						},
						{
							label: "This month",
							value: "thisMonth",
							range: {
								from: new Date(
									new Date().getFullYear(),
									new Date().getMonth(),
									1,
								),
								to: new Date(),
							},
						},
						{
							label: "Last month",
							value: "lastMonth",
							range: {
								from: new Date(
									new Date().getFullYear(),
									new Date().getMonth() - 1,
									1,
								),
								to: new Date(
									new Date().getFullYear(),
									new Date().getMonth(),
									0,
								),
							},
						},
					]}
					onPresetSelect={(preset) => {
						const presetRange = [
							{
								label: "Last 7 days",
								value: "7d",
								range: {
									from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
									to: new Date(),
								},
							},
							{
								label: "Last 30 days",
								value: "30d",
								range: {
									from: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
									to: new Date(),
								},
							},
							{
								label: "Last 90 days",
								value: "90d",
								range: {
									from: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
									to: new Date(),
								},
							},
							{
								label: "This month",
								value: "thisMonth",
								range: {
									from: new Date(
										new Date().getFullYear(),
										new Date().getMonth(),
										1,
									),
									to: new Date(),
								},
							},
							{
								label: "Last month",
								value: "lastMonth",
								range: {
									from: new Date(
										new Date().getFullYear(),
										new Date().getMonth() - 1,
										1,
									),
									to: new Date(
										new Date().getFullYear(),
										new Date().getMonth(),
										0,
									),
								},
							},
						].find((p) => p.value === preset);

						if (presetRange) {
							onDateRangeChange(presetRange.range);
						}
					}}
				/>

				<Button
					variant="outline"
					size="sm"
					onClick={onExport}
					className="flex w-full items-center justify-center gap-2 bg-transparent sm:w-auto"
				>
					<Download className="h-4 w-4" />
					<span className="sm:inline">Export</span>
				</Button>
			</div>
		</div>
	);
}
