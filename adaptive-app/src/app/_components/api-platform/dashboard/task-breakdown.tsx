"use client";

import { ChevronRight, Code, HelpCircle, FileText, Globe, Circle, BarChart, Send } from "lucide-react";
import { TaskBreakdownSkeleton } from "./loading-skeleton";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";

interface TaskBreakdownProps {
  data: DashboardData | null;
  loading: boolean;
  selectedProvider: string;
  providers: Provider[];
}

export function TaskBreakdown({
  data,
  loading,
  selectedProvider,
  providers,
}: TaskBreakdownProps) {
  if (loading) {
    return <TaskBreakdownSkeleton />;
  }

  if (!data) return null;

  const currentProvider = providers.find((p) => p.id === selectedProvider);

  return (
    <div className="bg-white dark:bg-[#0F0F12] rounded-xl p-6 border border-gray-200 dark:border-[#1F1F23]">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Task Type Performance
        </h2>
        <button className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300">
          View detailed breakdown
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {data.taskBreakdown.map((task, index) => (
          <div
            key={index}
            className="border border-gray-200 dark:border-[#1F1F23] rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <img src={task.icon} alt={task.name} className="w-5 h-5" />
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                    {task.name}
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </h3>
                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                    <span className="flex items-center gap-1"><Circle className="w-3 h-3 text-blue-500 fill-current" /> {task.requests} requests</span>
                    {task.inputTokens !== "0" && (
                      <span className="flex items-center gap-1"><BarChart className="w-3 h-3 text-gray-500" /> {task.inputTokens} input tokens</span>
                    )}
                    {task.outputTokens !== "0" && (
                      <span className="flex items-center gap-1"><Send className="w-3 h-3 text-gray-500" /> {task.outputTokens} output tokens</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {task.cost}
                </div>
                <div className="text-sm text-green-600 dark:text-green-400">
                  {task.savings} saved ({task.percentage}%)
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">
                  {currentProvider?.name || "Single provider"} cost
                </span>
                <span className="font-medium text-gray-500 dark:text-gray-400 line-through">
                  {task.comparisonCost}
                </span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-600 rounded-full transition-all duration-300"
                  style={{ width: `${100 - task.percentage}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
