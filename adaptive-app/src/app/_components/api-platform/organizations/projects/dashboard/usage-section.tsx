"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/trpc/react";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";
import { UsageChart } from "./charts/usage-chart";
import { ChartSkeleton } from "./loading-skeleton";
import { ModelSelector } from "./model-selector";

interface UsageSectionProps {
	data: DashboardData | null;
	loading: boolean;
	selectedProvider: string;
	providers: Provider[];
	selectedModel: string;
	onModelChange: (model: string) => void;
}

// Calculate direct cost for a specific model using actual token usage
function calculateDirectModelCost(
	usageData: { inputTokens: number; outputTokens: number }[],
	modelId: string,
	pricingData:
		| Record<string, { inputCost: number; outputCost: number }>
		| undefined,
): number {
	if (!pricingData || !pricingData[modelId]) {
		console.warn(`No pricing data found for model: ${modelId}`);
		return 0;
	}

	const modelPricing = pricingData[modelId];

	return usageData.reduce((totalCost, usage) => {
		const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputCost;
		const outputCost =
			(usage.outputTokens / 1_000_000) * modelPricing.outputCost;
		return totalCost + inputCost + outputCost;
	}, 0);
}

export function UsageSection({
	data,
	loading,
	selectedProvider: _selectedProvider,
	providers: _providers,
	selectedModel,
	onModelChange,
}: UsageSectionProps) {
	// Fetch dynamic pricing data
	const { data: modelPricing, isLoading: pricingLoading } =
		api.modelPricing.getAllModelPricing.useQuery();

	if (loading || pricingLoading) {
		return <ChartSkeleton />;
	}

	if (!data || !data.usageData || data.usageData.length === 0) {
		return (
			<Card>
				<CardHeader>
					<CardTitle>Usage vs Single Provider Cost</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No usage data available
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
		data.usageData,
		selectedModel,
		modelPricing,
	);

	const totalSavings = directModelCost - totalSpend;
	const savingsPercentage =
		directModelCost > 0
			? ((totalSavings / directModelCost) * 100).toFixed(1)
			: "0.0";

	// Recalculate chart data with selected model pricing
	const chartData = data.usageData.map((dataPoint) => {
		const selectedModelPricing = modelPricing?.[selectedModel];
		if (!selectedModelPricing) return dataPoint;

		const directCost =
			(dataPoint.inputTokens / 1_000_000) * selectedModelPricing.inputCost +
			(dataPoint.outputTokens / 1_000_000) * selectedModelPricing.outputCost;

		return {
			...dataPoint,
			singleProvider: directCost, // Update with selected model's cost
		};
	});

	return (
		<Card>
			<CardHeader>
				<div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
					<CardTitle>Cost Comparison</CardTitle>
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
			</CardHeader>
			<CardContent>
				<UsageChart data={chartData} providerName={selectedModelInfo?.name} />
			</CardContent>
		</Card>
	);
}
