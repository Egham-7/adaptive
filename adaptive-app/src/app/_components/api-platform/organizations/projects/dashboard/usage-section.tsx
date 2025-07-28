"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";
import { UsageChart } from "./charts/usage-chart";
import { ChartSkeleton } from "./loading-skeleton";

interface UsageSectionProps {
	data: DashboardData | null;
	loading: boolean;
	selectedProvider: string;
	providers: Provider[];
	selectedModel: string;
	onModelChange: (model: string) => void;
}

// Model pricing data (from seed-providers.ts)
const MODEL_PRICING = {
	"gpt-4o": { inputCost: 3.0, outputCost: 10.0 },
	"gpt-4o-mini": { inputCost: 0.15, outputCost: 0.6 },
	"claude-3.5-sonnet": { inputCost: 3.0, outputCost: 15.0 },
	"gemini-2.5-pro": { inputCost: 1.25, outputCost: 10.0 },
	"deepseek-chat": { inputCost: 0.27, outputCost: 1.1 },
} as const;

// Popular models for comparison
const COMPARISON_MODELS = [
	{ id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
	{ id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
	{ id: "claude-3.5-sonnet", name: "Claude 3.5 Sonnet", provider: "Anthropic" },
	{ id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "Google" },
	{ id: "deepseek-chat", name: "DeepSeek Chat", provider: "DeepSeek" },
] as const;

// Calculate direct cost for a specific model using actual token usage
function calculateDirectModelCost(
	usageData: { inputTokens: number; outputTokens: number }[],
	modelId: keyof typeof MODEL_PRICING
): number {
	const modelPricing = MODEL_PRICING[modelId];
	if (!modelPricing) return 0;

	return usageData.reduce((totalCost, usage) => {
		const inputCost = (usage.inputTokens / 1_000_000) * modelPricing.inputCost;
		const outputCost = (usage.outputTokens / 1_000_000) * modelPricing.outputCost;
		return totalCost + inputCost + outputCost;
	}, 0);
}

export function UsageSection({
	data,
	loading,
	selectedProvider,
	providers,
	selectedModel,
	onModelChange,
}: UsageSectionProps) {

	if (loading) {
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

	const selectedModelInfo = COMPARISON_MODELS.find((m) => m.id === selectedModel);
	const totalSpend = data.totalSpend;
	
	// Calculate actual direct model cost using real token usage data
	const directModelCost = calculateDirectModelCost(
		data.usageData, 
		selectedModel as keyof typeof MODEL_PRICING
	);
	
	const totalSavings = directModelCost - totalSpend;
	const savingsPercentage =
		directModelCost > 0
			? ((totalSavings / directModelCost) * 100).toFixed(1)
			: "0.0";

	// Recalculate chart data with selected model pricing
	const chartData = data.usageData.map((dataPoint) => {
		const modelPricing = MODEL_PRICING[selectedModel as keyof typeof MODEL_PRICING];
		if (!modelPricing) return dataPoint;

		const directCost = 
			(dataPoint.inputTokens / 1_000_000) * modelPricing.inputCost +
			(dataPoint.outputTokens / 1_000_000) * modelPricing.outputCost;

		return {
			...dataPoint,
			singleProvider: directCost, // Update with selected model's cost
		};
	});

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<div className="flex-1">
						<div className="mb-1 flex items-center justify-between">
							<CardTitle>Cost Comparison</CardTitle>
							<div className="flex items-center gap-2">
								<span className="text-muted-foreground text-sm">Compare vs</span>
								<Select value={selectedModel} onValueChange={onModelChange}>
									<SelectTrigger className="w-[180px]">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										{COMPARISON_MODELS.map((model) => (
											<SelectItem key={model.id} value={model.id}>
												{model.name}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							</div>
						</div>
						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<span className="text-muted-foreground text-sm">Your Adaptive Cost</span>
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
									Direct {selectedModelInfo?.name} Cost
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
								<span className="font-medium text-sm text-green-700 dark:text-green-400">
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

					</div>
				</div>
			</CardHeader>
			<CardContent>
				<UsageChart
					data={chartData}
					providerName={selectedModelInfo?.name}
				/>
			</CardContent>
		</Card>
	);
}
