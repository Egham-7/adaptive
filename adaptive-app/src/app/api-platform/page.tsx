"use client";

import { useState } from "react";
import { DashboardHeader } from "../_components/api-platform/dashboard/dashboard-header";
import { MetricsOverview } from "../_components/api-platform/dashboard/metrics-overview";
import { TaskBreakdown } from "../_components/api-platform/dashboard/task-breakdown";
import { UsageSection } from "../_components/api-platform/dashboard/usage-section";
import { useDateRange } from "@/hooks/use-date-range";
import { useDashboardData } from "@/hooks/api-platform/hooks/use-dashboard-data";
import type { DashboardFilters } from "@/types/api-platform/dashboard";

export default function DashboardPage() {
  const { dateRange, setDateRange } = useDateRange();
  const [selectedProvider, setSelectedProvider] = useState("openai-gpt4");

  const filters: DashboardFilters = {
    dateRange,
    provider: selectedProvider,
  };

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
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Failed to load dashboard data
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={refresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
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

      <MetricsOverview data={data} loading={loading} />

      <UsageSection
        data={data}
        loading={loading}
        selectedProvider={selectedProvider}
        providers={data?.providers || []}
      />

      <TaskBreakdown
        data={data}
        loading={loading}
        selectedProvider={selectedProvider}
        providers={data?.providers || []}
      />
    </div>
  );
}
