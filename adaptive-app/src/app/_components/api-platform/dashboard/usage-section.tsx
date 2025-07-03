"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { UsageChart } from "./charts/usage-chart";
import { TokenChart } from "./charts/token-chart";
import { RequestsChart } from "./charts/request-chart";
import { ChartSkeleton } from "./loading-skeleton";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";

interface UsageSectionProps {
  data: DashboardData | null;
  loading: boolean;
  selectedProvider: string;
  providers: Provider[];
}

export function UsageSection({
  data,
  loading,
  selectedProvider,
  providers,
}: UsageSectionProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        <div className="xl:col-span-3">
          <ChartSkeleton />
        </div>
        <div className="space-y-4">
          <ChartSkeleton height="120px" />
          <ChartSkeleton height="120px" />
          <ChartSkeleton height="120px" />
        </div>
      </div>
    );
  }

  if (!data) return null;

  const currentProvider = providers.find((p) => p.id === selectedProvider);
  const totalSpend = data.totalSpend;
  const totalComparison = currentProvider?.comparisonCosts.single || 0;
  const totalSavings = totalComparison - totalSpend;
  const savingsPercentage = ((totalSavings / totalComparison) * 100).toFixed(1);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
      {/* Main Usage Chart */}
      <div className="xl:col-span-3">
        <div className="bg-white dark:bg-[#0F0F12] rounded-xl p-6 border border-gray-200 dark:border-[#1F1F23]">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                Total Spend
              </h2>
              <div className="flex items-baseline gap-4">
                <span className="text-3xl font-bold text-gray-900 dark:text-white">
                  ${totalSpend.toFixed(2)}
                </span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  vs ${totalComparison.toFixed(2)} ({currentProvider?.name})
                </span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  ${totalSavings.toFixed(2)} saved ({savingsPercentage}%)
                </span>
              </div>
            </div>
            <div className="text-right">
              <Select defaultValue="1d">
                <SelectTrigger className="w-20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1d">1d</SelectItem>
                  <SelectItem value="7d">7d</SelectItem>
                  <SelectItem value="30d">30d</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <UsageChart
            data={data.usageData}
            providerName={currentProvider?.name}
          />
        </div>
      </div>

      {/* Right Sidebar Stats */}
      <div className="space-y-4">
        {/* Budget Card */}
        <div className="bg-white dark:bg-[#0F0F12] rounded-xl p-4 border border-gray-200 dark:border-[#1F1F23]">
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
            Monthly budget
          </h3>
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            ${totalSpend.toFixed(2)} / $50.00
          </div>
          <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            Resets in 7 days.{" "}
            <button className="text-blue-600 dark:text-blue-400 hover:underline">
              Edit budget
            </button>
          </div>
        </div>

        {/* Total Tokens */}
        <div className="bg-white dark:bg-[#0F0F12] rounded-xl p-4 border border-gray-200 dark:border-[#1F1F23]">
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
            Total tokens
          </h3>
          <div className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            {data.totalTokens.toLocaleString()}
          </div>
          <TokenChart data={data.tokenData} />
        </div>

        {/* Total Requests */}
        <div className="bg-white dark:bg-[#0F0F12] rounded-xl p-4 border border-gray-200 dark:border-[#1F1F23]">
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
            Total requests
          </h3>
          <div className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            {data.totalRequests.toLocaleString()}
          </div>
          <RequestsChart data={data.requestData} />
        </div>
      </div>
    </div>
  );
}
