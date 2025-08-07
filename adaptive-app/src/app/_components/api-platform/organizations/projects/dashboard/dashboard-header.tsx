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
		<div className="flex items-center justify-between">
			<h1 className="font-bold text-2xl text-foreground">Usage Dashboard</h1>
			<div className="flex items-center gap-3">

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
					className="flex items-center gap-2 bg-transparent"
				>
					<Download className="h-4 w-4" />
					Export
				</Button>
			</div>
		</div>
	);
}
