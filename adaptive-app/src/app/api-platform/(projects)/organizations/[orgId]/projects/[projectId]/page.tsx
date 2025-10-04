"use client";

import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { CostComparison } from "@/app/_components/api-platform/organizations/projects/dashboard/cost-comparison";
import { DashboardHeader } from "@/app/_components/api-platform/organizations/projects/dashboard/dashboard-header";
import { MetricsOverview } from "@/app/_components/api-platform/organizations/projects/dashboard/metrics-overview";
import { ProjectBreadcrumb } from "@/components/project-breadcrumb";
import { Button } from "@/components/ui/button";
import { useProjectDashboardData } from "@/hooks/usage/use-project-dashboard-data";
import { useDateRange } from "@/hooks/use-date-range";
import { setLastProject } from "@/hooks/use-smart-redirect";
import type { DashboardFilters } from "@/types/api-platform/dashboard";

export default function DashboardPage() {
	const params = useParams();
	const orgId = params.orgId as string;
	const projectId = params.projectId as string;
	const { dateRange, setDateRange } = useDateRange();
	const [selectedModels, setSelectedModels] = useState<string[]>([
		"openai:gpt-4o",
	]);

	const filters: DashboardFilters = useMemo(
		() => ({
			dateRange,
			provider: "all",
		}),
		[dateRange],
	);

	const { data, loading, error } = useProjectDashboardData(projectId, filters);

	// Track project visit for smart redirect
	useEffect(() => {
		if (orgId && projectId) {
			setLastProject(orgId, projectId);
		}
	}, [orgId, projectId]);

	const handleExport = () => {
		if (!data) return;

		const exportData = {
			dateRange,
			provider: "all",
			metrics: {
				totalSpend: data.totalSpend,
				totalSavings: data.totalSavings,
				savingsPercentage: data.totalSavingsPercentage,
				totalTokens: data.totalTokens,
				totalRequests: data.totalRequests,
			},
			dailyTrends: data.dailyTrends,
			requestTypeBreakdown: data.requestTypeBreakdown,
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
		<div className="w-full px-6 py-2">
			<div className="mb-6">
				<ProjectBreadcrumb />
			</div>
			{/* Header Section */}
			<div className="mb-8" id="dashboard-header">
				<DashboardHeader
					dateRange={dateRange}
					onDateRangeChange={setDateRange}
					onExport={handleExport}
				/>
			</div>

			{/* Main Dashboard Grid */}
			<div className="space-y-8">
				{/* Key Metrics Section */}
				<section className="space-y-6" id="dashboard-metrics">
					<h2 className="font-semibold text-2xl text-foreground">
						Key Performance Metrics
					</h2>
					<p className="text-muted-foreground">
						Real-time insights into your API usage and costs
					</p>
					<MetricsOverview data={data ?? null} loading={loading} />
				</section>

				{/* Provider Comparison Section */}
				<section className="space-y-6" id="dashboard-cost-comparison">
					<h2 className="font-semibold text-2xl text-foreground">
						Cost Comparison
					</h2>
					<p className="text-muted-foreground">
						Compare costs and performance across all models and providers
					</p>
					<div className="rounded-lg border bg-card p-6 shadow-sm">
						<CostComparison
							data={data ?? null}
							loading={loading}
							selectedModels={selectedModels}
							onModelsChange={setSelectedModels}
						/>
					</div>
				</section>
			</div>
		</div>
	);
}
