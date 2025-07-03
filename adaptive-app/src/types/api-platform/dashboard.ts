import type { DateRange } from "react-day-picker";

export interface DashboardData {
  totalSpend: number;
  totalSavings: number;
  savingsPercentage: number;
  totalTokens: number;
  totalRequests: number;
  usageData: UsageDataPoint[];
  tokenData: TokenDataPoint[];
  requestData: RequestDataPoint[];
  taskBreakdown: TaskBreakdownItem[];
  providers: Provider[];
}

export interface UsageDataPoint {
  date: string;
  adaptive: number;
  singleProvider: number;
  requests: number;
}

export interface TokenDataPoint {
  date: string;
  tokens: number;
}

export interface RequestDataPoint {
  date: string;
  requests: number;
}

export interface TaskBreakdownItem {
  id: string;
  name: string;
  icon: string;
  requests: string;
  inputTokens: string;
  outputTokens: string;
  cost: string;
  comparisonCost: string;
  savings: string;
  percentage: number;
}

export interface Provider {
  id: string;
  name: string;
  icon: string;
  comparisonCosts: {
    adaptive: number;
    single: number;
  };
}

export interface DashboardFilters {
  dateRange: DateRange | undefined;
  provider: string;
  refreshInterval?: number;
}

export interface MetricCardData {
  title: string;
  value: string;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon?: React.ReactNode;
  description?: string;
}
