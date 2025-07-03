"use client";

import { TrendingUp, TrendingDown } from "lucide-react";
import { LineChart, Line, ResponsiveContainer } from "recharts";

interface MetricChartCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon?: React.ReactNode;
  description?: string;
  data: Array<{ value: number }>;
  className?: string;
}

export function MetricChartCard({
  title,
  value,
  change,
  changeType = "neutral",
  icon,
  description,
  data,
  className = "",
}: MetricChartCardProps) {
  const getChangeIcon = () => {
    if (changeType === "positive") return <TrendingUp className="w-4 h-4" />;
    if (changeType === "negative") return <TrendingDown className="w-4 h-4" />;
    return null;
  };

  const getChangeColor = () => {
    if (changeType === "positive") return "text-green-600 dark:text-green-400";
    if (changeType === "negative") return "text-red-600 dark:text-red-400";
    return "text-gray-600 dark:text-gray-400";
  };

  const getLineColor = () => {
    if (changeType === "positive") return "#10b981";
    if (changeType === "negative") return "#ef4444";
    return "#6b7280";
  };

  return (
    <div
      className={`bg-white dark:bg-[#0F0F12] rounded-xl p-6 border border-gray-200 dark:border-[#1F1F23] hover:shadow-lg transition-shadow ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        {icon && (
          <div className="p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
            {icon}
          </div>
        )}
        {change && (
          <div
            className={`flex items-center gap-1 text-sm font-medium ${getChangeColor()}`}
          >
            {getChangeIcon()}
            {change}
          </div>
        )}
      </div>

      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {title}
          </h3>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {value}
          </p>
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-500">
              {description}
            </p>
          )}
        </div>

        <div className="h-16">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <Line
                type="monotone"
                dataKey="value"
                stroke={getLineColor()}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: getLineColor() }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}