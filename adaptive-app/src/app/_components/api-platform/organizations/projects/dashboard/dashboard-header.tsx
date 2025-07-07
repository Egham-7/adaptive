import { Download, RefreshCw } from "lucide-react";
import Image from "next/image";
import type { DateRange } from "react-day-picker";
import { Button } from "@/components/ui/button";
import { DateRangePicker } from "@/components/ui/date-range-picker";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import type { Provider, ProviderFilter } from "@/types/api-platform/dashboard";

interface DashboardHeaderProps {
	dateRange: DateRange;
	onDateRangeChange: (range: DateRange) => void;
	selectedProvider: ProviderFilter;
	onProviderChange: (provider: ProviderFilter) => void;
	providers: Provider[];
	onRefresh: () => void;
	onExport: () => void;
	isLoading?: boolean;
}

export function DashboardHeader({
	dateRange,
	onDateRangeChange,
	selectedProvider,
	onProviderChange,
	providers,
	onRefresh,
	onExport,
	isLoading = false,
}: DashboardHeaderProps) {
	return (
		<div className="flex items-center justify-between">
			<h1 className="font-bold text-2xl text-foreground">Usage Dashboard</h1>
			<div className="flex items-center gap-3">
				<Select value={selectedProvider} onValueChange={onProviderChange}>
					<SelectTrigger className="w-56">
						<SelectValue>
							{(() => {
								const selected = providers.find(
									(p) => p.id === selectedProvider,
								);
								return selected ? (
									<div className="flex items-center gap-2">
										<Image
											width={20}
											height={20}
											src={selected.icon}
											alt={selected.name}
										/>
										<span>{selected.name}</span>
									</div>
								) : null;
							})()}
						</SelectValue>
					</SelectTrigger>
					<SelectContent>
						{providers.map((provider) => (
							<SelectItem key={provider.id} value={provider.id}>
								<div className="flex items-center gap-2">
									<Image
										width={20}
										height={20}
										src={provider.icon}
										alt={provider.name}
									/>
									<span>{provider.name}</span>
								</div>
							</SelectItem>
						))}
					</SelectContent>
				</Select>

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
					onClick={onRefresh}
					disabled={isLoading}
					className="flex items-center gap-2 bg-transparent"
				>
					<RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
					Refresh
				</Button>

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
