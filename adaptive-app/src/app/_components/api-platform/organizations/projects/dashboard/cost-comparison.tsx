"use client";

import { BarChart3, Table } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/trpc/react";
import type { ProjectAnalytics } from "@/types/api-platform/dashboard";
import { UsageChart } from "./charts/usage-chart";
import { ChartSkeleton } from "./loading-skeleton";
import { ModelSelector } from "./model-selector";
import { ProviderComparisonTable } from "./provider-comparison-table";

interface CostComparisonProps {
	data: ProjectAnalytics | null;
	loading: boolean;
	selectedModel: string;
	onModelChange: (model: string) => void;
}

type ViewMode = "chart" | "table";

// Calculate direct cost for a specific model using actual token usage
const calculateDirectModelCost = (
	usageData: { inputTokens: number; outputTokens: number }[],
	modelId: string,
	pricingData:
		| Record<string, { inputCost: number; outputCost: number }>
		| undefined,
): number | null => {
	if (!pricingData || !pricingData[modelId]) {
		return null; // Return null instead of 0 when pricing is unavailable
	}

	const modelPricing = pricingData[modelId];

	return usageData.reduce((totalCost, usage) => {
		const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputCost;
		const outputCost =
			(usage.outputTokens / 1_000_000) * modelPricing.outputCost;
		return totalCost + inputCost + outputCost;
	}, 0);
};

export function CostComparison({
	data,
	loading,
	selectedModel,
	onModelChange,
}: CostComparisonProps) {
	const [viewMode, setViewMode] = useState<ViewMode>("chart");

	// Fetch dynamic pricing data
	const { data: modelPricing, isLoading: pricingLoading } =
		api.modelPricing.getAllModelPricing.useQuery();

	if (loading || pricingLoading) {
		return <ChartSkeleton />;
	}

	if (!data || !data.dailyTrends || data.dailyTrends.length === 0) {
		return (
			<Card>
				<CardHeader>
					<div className="flex items-center justify-between">
						<CardTitle>Cost Comparison</CardTitle>
						<div className="flex items-center gap-2">
							<Button
								variant={viewMode === "chart" ? "default" : "outline"}
								size="sm"
								onClick={() => setViewMode("chart")}
								className="gap-2"
							>
								<BarChart3 className="h-4 w-4" />
								Chart
							</Button>
							<Button
								variant={viewMode === "table" ? "default" : "outline"}
								size="sm"
								onClick={() => setViewMode("table")}
								className="gap-2"
							>
								<Table className="h-4 w-4" />
								Table
							</Button>
						</div>
					</div>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No data available
					</div>
				</CardContent>
			</Card>
		);
	}

	// Get selected model info from pricing data
	const selectedModelInfo = modelPricing?.[selectedModel]
		? {
				id: selectedModel,
				name:
					selectedModel.charAt(0).toUpperCase() +
					selectedModel.slice(1).replace(/-/g, " "),
			}
		: null;
	const totalSpend = data.totalSpend;

	// Calculate actual direct model cost using real token usage data
	const directModelCost = calculateDirectModelCost(
		data.dailyTrends,
		selectedModel,
		modelPricing,
	);

	// Don't show comparison if we don't have pricing for the selected model
	if (directModelCost === null) {
		return (
			<Card>
				<CardHeader>
					<div className="flex items-center justify-between">
						<div className="flex items-center gap-3">
							<CardTitle>Cost Comparison</CardTitle>
							<Badge variant="secondary">No Pricing Data</Badge>
						</div>
						<div className="flex items-center gap-2">
							<Button
								variant={viewMode === "chart" ? "default" : "outline"}
								size="sm"
								onClick={() => setViewMode("chart")}
								className="gap-2"
							>
								<BarChart3 className="h-4 w-4" />
								Chart
							</Button>
							<Button
								variant={viewMode === "table" ? "default" : "outline"}
								size="sm"
								onClick={() => setViewMode("table")}
								className="gap-2"
							>
								<Table className="h-4 w-4" />
								Table
							</Button>
						</div>
					</div>
					<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
						<div>
							<p className="text-muted-foreground text-sm">
								Pricing data not available for selected model
							</p>
						</div>
						<ModelSelector
							selectedModel={selectedModel}
							onModelChange={onModelChange}
						/>
					</div>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						Select a model with available pricing data to see comparison
					</div>
				</CardContent>
			</Card>
		);
	}

	const totalSavings = directModelCost - totalSpend;
	const savingsPercentage =
		directModelCost > 0
			? ((totalSavings / directModelCost) * 100).toFixed(1)
			: "0.0";

	// Recalculate chart data with selected model pricing
	const chartData = data.dailyTrends.map((dataPoint) => {
		const selectedModelPricing = modelPricing?.[selectedModel];

		// Fallback to 0 cost if no pricing available
		const directCost = selectedModelPricing
			? (dataPoint.inputTokens / 1_000_000) * selectedModelPricing.inputCost +
				(dataPoint.outputTokens / 1_000_000) * selectedModelPricing.outputCost
			: 0;

		return {
			...dataPoint,
			adaptive: dataPoint.spend,
			singleProvider: directCost,
		};
	});

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-3">
						<CardTitle>Cost Comparison</CardTitle>
						<Badge variant="secondary">
							{viewMode === "chart" ? "Model Chart" : "Provider Table"}
						</Badge>
					</div>
					<div className="flex items-center gap-2">
						<Button
							variant={viewMode === "chart" ? "default" : "outline"}
							size="sm"
							onClick={() => setViewMode("chart")}
							className="gap-2"
						>
							<BarChart3 className="h-4 w-4" />
							Chart
						</Button>
						<Button
							variant={viewMode === "table" ? "default" : "outline"}
							size="sm"
							onClick={() => setViewMode("table")}
							className="gap-2"
						>
							<Table className="h-4 w-4" />
							Table
						</Button>
					</div>
				</div>

				{viewMode === "chart" && (
					<>
						<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
							<div>
								<p className="text-muted-foreground text-sm">
									Compare your Adaptive costs against specific models
								</p>
							</div>
							<ModelSelector
								selectedModel={selectedModel}
								onModelChange={onModelChange}
							/>
						</div>
						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<span className="text-muted-foreground text-sm">
									Your Adaptive Cost
								</span>
								<span className="font-semibold text-lg">
									$
									{typeof totalSpend === "number"
										? totalSpend < 0.01 && totalSpend > 0
											? totalSpend.toFixed(6)
											: totalSpend.toFixed(2)
										: totalSpend}
								</span>
							</div>
							<div className="flex items-center justify-between">
								<span className="text-muted-foreground text-sm">
									Direct {selectedModelInfo?.name || "Model"} Cost
								</span>
								<span className="font-semibold text-lg">
									$
									{typeof directModelCost === "number"
										? directModelCost < 0.01 && directModelCost > 0
											? directModelCost.toFixed(6)
											: directModelCost.toFixed(2)
										: directModelCost}
								</span>
							</div>
						</div>
						<div className="mt-3 rounded-lg bg-green-50 p-3 dark:bg-green-900/10">
							<div className="flex items-center justify-between">
								<span className="font-medium text-green-700 text-sm dark:text-green-400">
									You saved with Adaptive
								</span>
								<span className="font-bold text-green-700 dark:text-green-400">
									$
									{typeof totalSavings === "number"
										? totalSavings < 0.01 && totalSavings > 0
											? totalSavings.toFixed(6)
											: totalSavings.toFixed(2)
										: totalSavings}{" "}
									({savingsPercentage}%)
								</span>
							</div>
						</div>
					</>
				)}

				{viewMode === "table" && (
					<p className="text-muted-foreground text-sm">
						Compare your Adaptive costs against what you would pay using each
						provider exclusively
					</p>
				)}
			</CardHeader>
			<CardContent>
				{viewMode === "chart" ? (
					<UsageChart data={chartData} providerName={selectedModelInfo?.name} />
				) : (
					<ProviderComparisonTable
						data={data}
						loading={loading}
						selectedModel={selectedModel}
					/>
				)}
			</CardContent>
		</Card>
	);
}
