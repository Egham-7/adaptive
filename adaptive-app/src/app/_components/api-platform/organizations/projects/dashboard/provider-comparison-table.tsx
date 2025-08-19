"use client";

import { ArrowDown, ArrowUp } from "lucide-react";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import type { ProjectAnalytics } from "@/types/api-platform/dashboard";

const PROVIDER_ICONS = {
	openai: "/logos/openai.webp",
	anthropic: "/logos/anthropic.jpeg",
	gemini: "/logos/google.svg",
	groq: "/logos/groq.png",
	grok: "/logos/grok.svg",
	deepseek: "/logos/deepseek.svg",
	huggingface: "/logos/huggingface.png",
} as const;

interface ProviderComparisonTableProps {
	data: ProjectAnalytics | null;
	loading: boolean;
	selectedModel?: string;
}

export function ProviderComparisonTable({
	data,
	loading,
	selectedModel,
}: ProviderComparisonTableProps) {
	if (loading) {
		return (
			<Card>
				<CardContent className="p-6">
					<div className="animate-pulse">
						<div className="mb-4 h-4 w-1/3 rounded bg-muted" />
						<div className="space-y-3">
							{[...Array(5)].map((_, i) => (
								<div key={i} className="flex space-x-4">
									<div className="h-4 w-1/4 rounded bg-muted" />
									<div className="h-4 w-1/4 rounded bg-muted" />
									<div className="h-4 w-1/4 rounded bg-muted" />
									<div className="h-4 w-1/4 rounded bg-muted" />
								</div>
							))}
						</div>
					</div>
				</CardContent>
			</Card>
		);
	}

	if (
		!data ||
		!data.modelProviderBreakdown ||
		data.modelProviderBreakdown.length === 0
	) {
		return (
			<Card>
				<CardHeader>
					<div className="flex items-center justify-between">
						<CardTitle>Model Cost Comparison</CardTitle>
						<Badge variant="secondary">
							{selectedModel
								? selectedModel.charAt(0).toUpperCase() +
									selectedModel.slice(1).replace(/-/g, " ")
								: "All Models"}
						</Badge>
					</div>
					<p className="text-muted-foreground text-sm">
						Compare your Adaptive costs against specific models by provider
					</p>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No model comparison data available
					</div>
				</CardContent>
			</Card>
		);
	}

	// Use the model-provider breakdown directly from tRPC
	const tableData = data.modelProviderBreakdown.map((item) => ({
		id: `${item.model}-${item.provider}`,
		modelName:
			item.model.charAt(0).toUpperCase() +
			item.model.slice(1).replace(/-/g, " "),
		providerName:
			item.provider.charAt(0).toUpperCase() + item.provider.slice(1),
		icon:
			PROVIDER_ICONS[item.provider as keyof typeof PROVIDER_ICONS] ||
			"/logos/default.svg",
		adaptiveCost: data.totalSpend,
		modelCost: item.estimatedCost,
		savings: item.savings,
		savingsPercentage: item.savingsPercentage,
	}));

	const sortedData = [...tableData].sort((a, b) => b.modelCost - a.modelCost);

	const formatCurrency = (amount: number) => {
		return new Intl.NumberFormat("en-US", {
			style: "currency",
			currency: "USD",
			minimumFractionDigits: 2,
		}).format(amount);
	};

	const _calculateSavings = (adaptive: number, modelCost: number) => {
		const savings = modelCost - adaptive;
		const percentage = modelCost > 0 ? (savings / modelCost) * 100 : 0;
		return { amount: savings, percentage };
	};

	return (
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<CardTitle>Provider Cost Comparison</CardTitle>
					<Badge variant="secondary">All Providers</Badge>
				</div>
				<p className="text-muted-foreground text-sm">
					Compare your Adaptive costs against what you would pay using each
					provider exclusively
				</p>
			</CardHeader>
			<CardContent>
				<div className="rounded-md border">
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Model</TableHead>
								<TableHead>Provider</TableHead>
								<TableHead className="text-right">Your Cost</TableHead>
								<TableHead className="text-right">Model Cost</TableHead>
								<TableHead className="text-right">Savings</TableHead>
								<TableHead className="text-right">% Saved</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{sortedData.map((item) => {
								const isPositiveSavings = item.savings > 0;

								return (
									<TableRow key={item.id}>
										<TableCell>
											<span className="font-medium">{item.modelName}</span>
										</TableCell>
										<TableCell>
											<div className="flex items-center gap-3">
												<Image
													width={20}
													height={20}
													src={item.icon}
													alt={item.providerName}
													className="rounded"
												/>
												<span className="text-muted-foreground text-sm">
													{item.providerName}
												</span>
											</div>
										</TableCell>
										<TableCell className="text-right font-mono">
											{formatCurrency(item.adaptiveCost)}
										</TableCell>
										<TableCell className="text-right font-mono">
											{formatCurrency(item.modelCost)}
										</TableCell>
										<TableCell className="text-right font-mono">
											<div className="flex items-center justify-end gap-1">
												{isPositiveSavings ? (
													<ArrowDown className="h-3 w-3 text-green-600" />
												) : (
													<ArrowUp className="h-3 w-3 text-red-600" />
												)}
												<span
													className={
														isPositiveSavings
															? "text-green-600"
															: "text-red-600"
													}
												>
													{formatCurrency(Math.abs(item.savings))}
												</span>
											</div>
										</TableCell>
										<TableCell className="text-right">
											<Badge
												variant={isPositiveSavings ? "default" : "destructive"}
												className={
													isPositiveSavings
														? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
														: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
												}
											>
												{isPositiveSavings ? "+" : ""}
												{item.savingsPercentage.toFixed(1)}%
											</Badge>
										</TableCell>
									</TableRow>
								);
							})}
						</TableBody>
					</Table>
				</div>

				<div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
					<div className="rounded-lg border p-4">
						<div className="text-muted-foreground text-sm">Average Savings</div>
						<div className="font-bold text-2xl text-green-600">
							{formatCurrency(
								sortedData.reduce(
									(total, item) => total + Math.max(0, item.savings),
									0,
								) / sortedData.length,
							)}
						</div>
					</div>
					<div className="rounded-lg border p-4">
						<div className="text-muted-foreground text-sm">Best Savings %</div>
						<div className="font-bold text-2xl text-green-600">
							{Math.max(
								...sortedData.map((item) => item.savingsPercentage),
							).toFixed(1)}
							%
						</div>
					</div>
					<div className="rounded-lg border p-4">
						<div className="text-muted-foreground text-sm">
							Your Total Spend
						</div>
						<div className="font-bold text-2xl">
							{formatCurrency(data.totalSpend)}
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
