"use client";

import { DollarSign, Zap, GitBranch, Shield } from "lucide-react";
import { MetricCard } from "./metric-card";
import { MetricChartCard } from "./metric-chart-card";
import { MetricCardSkeleton } from "./loading-skeleton";
import type { DashboardData } from "@/types/api-platform/dashboard";

interface MetricsOverviewProps {
  data: DashboardData | null;
  loading: boolean;
}

export function MetricsOverview({ data, loading }: MetricsOverviewProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <MetricCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (!data) return null;

  const scalarMetrics = [
    {
      title: "Total Cost Savings",
      value: `$${data.totalSavings.toFixed(2)}`,
      change: `+${data.savingsPercentage.toFixed(1)}%`,
      changeType: "positive" as const,
      icon: <DollarSign className="w-5 h-5 text-green-600" />,
      description: "vs single provider",
    },
    {
      title: "Total Spend",
      value: `$${data.totalSpend.toFixed(2)}`,
      change: "-31.2%",
      changeType: "positive" as const,
      icon: <Zap className="w-5 h-5 text-blue-600" />,
      description: "current period",
    },
  ];

  const chartMetrics = [
    {
      title: "Total Tokens",
      value: data.totalTokens.toLocaleString(),
      change: "+12.5%",
      changeType: "positive" as const,
      icon: <GitBranch className="w-5 h-5 text-purple-600" />,
      description: "processed",
      data: data.tokenData.map(d => ({ value: d.tokens })),
    },
    {
      title: "Total Requests",
      value: data.totalRequests.toLocaleString(),
      change: "+8.3%",
      changeType: "positive" as const,
      icon: <Shield className="w-5 h-5 text-orange-600" />,
      description: "completed",
      data: data.requestData.map(d => ({ value: d.requests })),
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
      {scalarMetrics.map((metric, index) => (
        <MetricCard key={index} {...metric} />
      ))}
      {chartMetrics.map((metric, index) => (
        <MetricChartCard key={`chart-${index}`} {...metric} />
      ))}
    </div>
  );
}
