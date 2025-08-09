"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { DashboardHeader } from "@/app/_components/api-platform/organizations/projects/dashboard/dashboard-header";
import { MetricsOverview } from "@/app/_components/api-platform/organizations/projects/dashboard/metrics-overview";
import { ProviderComparisonTable } from "@/app/_components/api-platform/organizations/projects/dashboard/provider-comparison-table";
import { UsageSection } from "@/app/_components/api-platform/organizations/projects/dashboard/usage-section";
import { Button } from "@/components/ui/button";
import { useProjectDashboardData } from "@/hooks/usage/use-project-dashboard-data";
import { useDateRange } from "@/hooks/use-date-range";
import type { DashboardFilters } from "@/types/api-platform/dashboard";

export default function DashboardPage() {
  const params = useParams();
  const orgId = params.orgId as string;
  const projectId = params.projectId as string;
  const { dateRange, setDateRange } = useDateRange();
  const [selectedModel, setSelectedModel] = useState<string>("gpt-4o");

  const filters: DashboardFilters = useMemo(
    () => ({
      dateRange,
      provider: "all",
    }),
    [dateRange]
  );

  const { data, loading, error } = useProjectDashboardData(projectId, filters);

  const handleExport = () => {
    if (!data) return;

    const exportData = {
      dateRange,
      provider: "all",
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
    <div className="w-full px-6 py-2">
      {/* Back Navigation */}
      <div className="mb-6">
        <Link href={`/api-platform/organizations/${orgId}`}>
          <Button variant="ghost" size="sm">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Projects
          </Button>
        </Link>
      </div>

      {/* Header Section */}
      <div className="mb-8">
        <DashboardHeader
          dateRange={dateRange}
          onDateRangeChange={setDateRange}
          onExport={handleExport}
        />
      </div>

      {/* Main Dashboard Grid */}
      <div className="space-y-8">
        {/* Key Metrics Section */}
        <section className="space-y-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="font-semibold text-2xl text-foreground">
                Key Performance Metrics
              </h2>
              <p className="text-muted-foreground">
                Real-time insights into your API usage and costs
              </p>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <div className="h-2 w-2 rounded-full bg-green-500" />
              <span>Live data</span>
            </div>
          </div>
          <MetricsOverview
            data={data}
            loading={loading}
            selectedModel={selectedModel}
          />
        </section>

        {/* Analytics Grid Layout */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-4">
          {/* Main Analytics Area */}
          <div className="space-y-8 lg:col-span-4">
            {/* Provider Comparison Section */}
            <section className="space-y-6">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="font-semibold text-2xl text-foreground">
                    Cost Comparison
                  </h2>
                  <p className="text-muted-foreground">
                    Compare costs and performance across all models ands
                    providers
                  </p>
                </div>
                <div className="text-muted-foreground text-sm">
                  Based on current usage patterns
                </div>
              </div>
              <div className="rounded-lg border bg-card p-6 shadow-sm">
                <UsageSection
                  data={data}
                  loading={loading}
                  selectedProvider="all"
                  providers={data?.providers || []}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                />
              </div>
              <div className="rounded-lg border bg-card p-6 shadow-sm">
                <ProviderComparisonTable data={data} loading={loading} />
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
