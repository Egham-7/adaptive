"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { useDashboardData } from "@/hooks/api-platform/hooks/use-dashboard-data";
import { useDateRange } from "@/hooks/use-date-range";
import type { DashboardFilters } from "@/types/api-platform/dashboard";
import { DashboardHeader } from "../_components/api-platform/dashboard/dashboard-header";
import { MetricsOverview } from "../_components/api-platform/dashboard/metrics-overview";
import { TaskDistributionChart } from "../_components/api-platform/dashboard/task-distribution-chart";
import { UsageSection } from "../_components/api-platform/dashboard/usage-section";

export default function DashboardPage() {
	const { dateRange, setDateRange } = useDateRange();
	const [selectedProvider, setSelectedProvider] = useState("openai");

	const filters: DashboardFilters = useMemo(
		() => ({
			dateRange,
			provider: selectedProvider,
		}),
		[dateRange, selectedProvider],
	);

	const { data, loading, error, refresh } = useDashboardData(filters);

	const handleExport = () => {
		if (!data) return;

		const exportData = {
			dateRange,
			provider: selectedProvider,
			metrics: {
				totalSpend: data.totalSpend,
				totalSavings: data.totalSavings,
				savingsPercentage: data.savingsPercentage,
				totalTokens: data.totalTokens,
				totalRequests: data.totalRequests,
			},
			usageData: data.usageData,
			taskBreakdown: data.taskBreakdown,
		};

		const blob = new Blob([JSON.stringify(exportData, null, 2)], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `dashboard-export-${new Date().toISOString().split("T")[0]}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	};

	if (error) {
		return (
			<div className="flex min-h-[400px] items-center justify-center">
				<div className="text-center">
					<h3 className="mb-2 font-medium text-gray-900 text-lg dark:text-white">
						Failed to load dashboard data
					</h3>
					<p className="mb-4 text-gray-600 dark:text-gray-400">{error}</p>
					<Button onClick={refresh}>Try Again</Button>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-8">
			{/* Header Section */}
			<DashboardHeader
				dateRange={dateRange}
				onDateRangeChange={setDateRange}
				selectedProvider={selectedProvider}
				onProviderChange={setSelectedProvider}
				providers={data?.providers || []}
				onRefresh={refresh}
				onExport={handleExport}
				isLoading={loading}
			/>

			{/* Key Metrics Section */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<h2 className="font-semibold text-gray-900 text-xl dark:text-white">
						Key Performance Metrics
					</h2>
					<div className="text-gray-500 text-sm dark:text-gray-400">
						Real-time insights
					</div>
				</div>
				<MetricsOverview data={data} loading={loading} />
			</section>

			{/* Divider */}
			<div className="border-gray-200 border-t dark:border-gray-800" />

			{/* Usage Analytics Section */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<h2 className="font-semibold text-gray-900 text-xl dark:text-white">
						Usage Analytics
					</h2>
					<div className="text-gray-500 text-sm dark:text-gray-400">
						Spend trends over time
					</div>
				</div>
				<div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
					<div className="lg:col-span-2">
						<UsageSection
							data={data}
							loading={loading}
							selectedProvider={selectedProvider}
							providers={data?.providers || []}
						/>
					</div>
					<TaskDistributionChart data={data} loading={loading} />
				</div>
			</section>

			{/* Divider */}
			<div className="border-gray-200 border-t dark:border-gray-800" />
		</div>
	);
}
