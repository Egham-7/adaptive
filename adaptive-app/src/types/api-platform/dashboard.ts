import type { DateRange } from "react-day-picker";
import type { RouterInputs, RouterOutputs } from "../index";

// ---- tRPC-derived Types ----

/**
 * Project analytics data from the usage router
 */
export type ProjectAnalytics = RouterOutputs["usage"]["getProjectAnalytics"];

/**
 * User analytics data from the usage router
 */
export type UserAnalytics = RouterOutputs["usage"]["getUserAnalytics"];

/**
 * Daily usage trend data point
 */
export type DailyTrendDataPoint = ProjectAnalytics["dailyTrends"][number];

/**
 * Provider breakdown data point
 */
export type ProviderBreakdown = ProjectAnalytics["providerBreakdown"][number];

/**
 * Request type breakdown data point
 */
export type RequestTypeBreakdown =
	ProjectAnalytics["requestTypeBreakdown"][number];

// ---- Chart Data Types (UI-formatted from tRPC types) ----

/**
 * Data point for usage comparison charts
 * Note: date is formatted as string for UI display
 */
export interface UsageDataPoint {
	date: string;
	adaptive: number;
	singleProvider: number;
	requests: number;
	spend: number;
	tokens: number;
	inputTokens: number;    // ← Add input tokens for accurate cost calculations
	outputTokens: number;   // ← Add output tokens for accurate cost calculations
}

/**
 * Data point for token usage charts
 * Note: date is formatted as string for UI display
 */
export interface TokenDataPoint {
	date: string;
	tokens: number;
	spend: number;
	requests: number;
}

/**
 * Data point for request count charts
 * Note: date is formatted as string for UI display
 */
export interface RequestDataPoint {
	date: string;
	requests: number;
	spend: number;
	tokens: number;
}

/**
 * Data point for error rate charts
 * Note: date is formatted as string for UI display
 */
export interface ErrorRateDataPoint {
	date: string;
	errorRate: number;
	errorCount: number;
}

// ---- UI-specific Types ----

/**
 * Provider type from tRPC schema
 */
export type ProviderType =
	RouterInputs["usage"]["getProjectAnalytics"]["provider"];

/**
 * Supported provider types for filtering - includes "all" option
 */
export type ProviderFilter = "all" | NonNullable<ProviderType>;

/**
 * Complete dashboard data structure for UI components
 */
export interface DashboardData {
	totalSpend: number;
	totalSavings: number;
	savingsPercentage: number;
	totalTokens: number;
	totalRequests: number;
	errorRate: number;
	errorCount: number;
	usageData: UsageDataPoint[];
	tokenData: TokenDataPoint[];
	requestData: RequestDataPoint[];
	errorRateData: ErrorRateDataPoint[];
	taskBreakdown: TaskBreakdownItem[];
	providers: Provider[];
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
	provider: ProviderFilter;
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
