"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { DashboardHeader } from "@/app/_components/api-platform/organizations/projects/dashboard/dashboard-header";
import { MetricsOverview } from "@/app/_components/api-platform/organizations/projects/dashboard/metrics-overview";
import { TaskDistributionChart } from "@/app/_components/api-platform/organizations/projects/dashboard/task-distribution-chart";
import { UsageSection } from "@/app/_components/api-platform/organizations/projects/dashboard/usage-section";
import { Button } from "@/components/ui/button";
import { useProjectDashboardData } from "@/hooks/usage/use-project-dashboard-data";
import { useDateRange } from "@/hooks/use-date-range";
import type {
	DashboardFilters,
	ProviderFilter,
} from "@/types/api-platform/dashboard";

export default function DashboardPage() {
	const params = useParams();
	const orgId = params.orgId as string;
	const projectId = params.projectId as string;
	const { dateRange, setDateRange } = useDateRange();
	const [selectedProvider, setSelectedProvider] =
		useState<ProviderFilter>("all");
	const [selectedModel, setSelectedModel] = useState<string>("gpt-4o");

	const filters: DashboardFilters = useMemo(
		() => ({
			dateRange,
			provider: selectedProvider,
		}),
		[dateRange, selectedProvider],
	);

	const { data, loading, error } = useProjectDashboardData(projectId, filters);

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
					<h3 className="mb-2 font-medium text-foreground text-lg">
						Failed to load dashboard data
					</h3>
					<p className="mb-4 text-muted-foreground">{error}</p>
					<Button onClick={() => window.location.reload()}>Try Again</Button>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-8">
			{/* Back Navigation */}
			<div className="mb-4">
				<Link href={`/api-platform/organizations/${orgId}`}>
					<Button variant="ghost" size="sm">
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Projects
					</Button>
				</Link>
			</div>

			{/* Header Section */}
			<DashboardHeader
				dateRange={dateRange}
				onDateRangeChange={setDateRange}
				selectedProvider={selectedProvider}
				onProviderChange={setSelectedProvider}
				providers={data?.providers || []}
				onExport={handleExport}
			/>

			{/* Key Metrics Section */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<h2 className="font-semibold text-foreground text-xl">
						Key Performance Metrics
					</h2>
					<div className="text-muted-foreground text-sm">
						Real-time insights
					</div>
				</div>
				<MetricsOverview
					data={data}
					loading={loading}
					selectedModel={selectedModel}
				/>
			</section>

			{/* Divider */}
			<div className="border-border border-t" />

			{/* Usage Analytics Section */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<h2 className="font-semibold text-foreground text-xl">
						Usage Analytics
					</h2>
					<div className="text-muted-foreground text-sm">
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
							selectedModel={selectedModel}
							onModelChange={setSelectedModel}
						/>
					</div>
					<TaskDistributionChart data={data} loading={loading} />
				</div>
			</section>

			{/* Divider */}
			<div className="border-border border-t" />
		</div>
	);
}
