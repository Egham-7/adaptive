"use client";

import {
  FaChartLine,
  FaCoins,
  FaDollarSign,
  FaExclamationTriangle,
  FaServer,
} from "react-icons/fa";
import type { DashboardData } from "@/types/api-platform/dashboard";
import { MetricCardSkeleton } from "./loading-skeleton";
import { VersatileMetricChart } from "./versatile-metric-chart";

// Model pricing data (same as usage-section.tsx)
const MODEL_PRICING = {
  "gpt-4o": { inputCost: 3.0, outputCost: 10.0 },
  "gpt-4o-mini": { inputCost: 0.15, outputCost: 0.6 },
  "claude-3.5-sonnet": { inputCost: 3.0, outputCost: 15.0 },
  "gemini-2.5-pro": { inputCost: 1.25, outputCost: 10.0 },
  "deepseek-chat": { inputCost: 0.27, outputCost: 1.1 },
} as const;

// Calculate direct cost for a specific model using actual token usage
function calculateDirectModelCost(
  usageData: { inputTokens: number; outputTokens: number }[],
  modelId: keyof typeof MODEL_PRICING
): number {
  const modelPricing = MODEL_PRICING[modelId];
  if (!modelPricing) return 0;

  return usageData.reduce((totalCost, usage) => {
    const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputCost;
    const outputCost =
      (usage.outputTokens / 1_000_000) * modelPricing.outputCost;
    return totalCost + inputCost + outputCost;
  }, 0);
}

interface MetricsOverviewProps {
  data: DashboardData | null;
  loading: boolean;
  selectedModel?: string;
}

export function MetricsOverview({
  data,
  loading,
  selectedModel = "gpt-4o",
}: MetricsOverviewProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: Using index for skeleton components is acceptable
          <MetricCardSkeleton key={`skeleton-${i}`} />
        ))}
      </div>
    );
  }

  if (!data) return null;

  // Calculate dynamic costs for the selected model
  const usageDataWithDynamicCosts = data.usageData.map((d) => {
    const adaptiveCost = d.adaptive;
    const modelCost = calculateDirectModelCost(
      [{ inputTokens: d.inputTokens || 0, outputTokens: d.outputTokens || 0 }],
      selectedModel as keyof typeof MODEL_PRICING
    );
    return {
      ...d,
      adaptiveCost,
      modelCost,
      savings: Math.max(0, modelCost - adaptiveCost),
    };
  });

  const savingsData = usageDataWithDynamicCosts.map((d) => ({
    date: d.date,
    value: d.savings,
  }));

  const spendData = data.usageData.map((d) => ({
    date: d.date,
    value: d.adaptive, // This is the actual customer spending (same as adaptive line in usage chart)
  }));

  const allMetrics = [
    {
      title: "Cost Savings Trend",
      chartType: "area" as const,
      icon: <FaDollarSign className="h-5 w-5 text-success" />,
      data: savingsData,
      color: "hsl(var(--chart-1))",
      totalValue: `$${(() => {
        const totalSavings = usageDataWithDynamicCosts.reduce(
          (sum, d) => sum + d.savings,
          0
        );
        const str = totalSavings.toString();
        const parts = str.split(".");
        const decimalPart = parts[1] || "";
        const significantDecimals = decimalPart.replace(/0+$/, "").length;
        const decimals = Math.min(Math.max(significantDecimals, 2), 8);
        return totalSavings.toFixed(decimals);
      })()}`,
    },
    {
      title: "Spending Over Time",
      chartType: "line" as const,
      icon: <FaChartLine className="h-5 w-5 text-chart-2" />,
      data: spendData,
      color: "hsl(var(--chart-2))",
      totalValue: `$${(() => {
        const str = data.totalSpend.toString();
        const parts = str.split(".");
        const decimalPart = parts[1] || "";
        const significantDecimals = decimalPart.replace(/0+$/, "").length;
        const decimals = Math.min(Math.max(significantDecimals, 2), 8);
        return data.totalSpend.toFixed(decimals);
      })()}`,
    },
    {
      title: "Token Usage",
      chartType: "bar" as const,
      icon: <FaCoins className="h-5 w-5 text-chart-3" />,
      data: data.tokenData.map((d) => ({ date: d.date, value: d.tokens })),
      color: "hsl(var(--chart-3))",
      totalValue: data.totalTokens.toLocaleString(),
    },
    {
      title: "Request Volume",
      chartType: "area" as const,
      icon: <FaServer className="h-5 w-5 text-chart-4" />,
      data: data.requestData.map((d) => ({ date: d.date, value: d.requests })),
      color: "hsl(var(--chart-4))",
      totalValue: data.totalRequests.toLocaleString(),
    },
    {
      title: "Error Rate",
      chartType: "area" as const,
      icon: <FaExclamationTriangle className="h-5 w-5 text-destructive" />,
      data: data.errorRateData.map((d) => ({
        date: d.date,
        value: d.errorRate,
      })),
      color: "hsl(var(--destructive))",
      totalValue: `${data.errorRate.toFixed(2)}%`,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
      {allMetrics.map((metric) => (
        <VersatileMetricChart
          key={metric.title}
          title={metric.title}
          chartType={metric.chartType}
          data={metric.data}
          icon={metric.icon}
          color={metric.color}
          totalValue={metric.totalValue}
        />
      ))}
    </div>
  );
}
