"use client";

import { ArrowDown, ArrowUp, Search } from "lucide-react";
import Image from "next/image";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
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
	const [searchTerm, setSearchTerm] = useState("");
	const [providerFilter, setProviderFilter] = useState<string>("all");
	const [sortBy, setSortBy] = useState<"savings" | "cost" | "model">("savings");
	const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

	// Use the model-provider breakdown directly from tRPC
	const tableData = useMemo(() => {
		if (!data?.modelProviderBreakdown) return [];

		let filtered = data.modelProviderBreakdown.map((item) => ({
			id: `${item.model}-${item.provider}`,
			modelName:
				item.model.charAt(0).toUpperCase() +
				item.model.slice(1).replace(/-/g, " "),
			providerName:
				item.provider.charAt(0).toUpperCase() + item.provider.slice(1),
			icon:
				PROVIDER_ICONS[item.provider as keyof typeof PROVIDER_ICONS] ||
				"/logos/default.svg",
			estimatedCost: item.estimatedCost,
			savings: item.savings,
			savingsPercentage: item.savingsPercentage,
			adaptiveCost: data.totalSpend / data.modelProviderBreakdown.length, // Distribute adaptive cost
		}));

		// Apply filters
		if (searchTerm) {
			filtered = filtered.filter(
				(item) =>
					item.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
					item.providerName.toLowerCase().includes(searchTerm.toLowerCase()),
			);
		}

		if (providerFilter !== "all") {
			filtered = filtered.filter(
				(item) =>
					item.providerName.toLowerCase() === providerFilter.toLowerCase(),
			);
		}

		// Apply sorting
		filtered.sort((a, b) => {
			let valueA: number | string;
			let valueB: number | string;

			switch (sortBy) {
				case "savings":
					valueA = a.savingsPercentage;
					valueB = b.savingsPercentage;
					break;
				case "cost":
					valueA = a.estimatedCost;
					valueB = b.estimatedCost;
					break;
				case "model":
					valueA = a.modelName;
					valueB = b.modelName;
					break;
				default:
					valueA = a.savingsPercentage;
					valueB = b.savingsPercentage;
			}

			if (sortOrder === "asc") {
				return valueA < valueB ? -1 : valueA > valueB ? 1 : 0;
			}
			return valueA > valueB ? -1 : valueA < valueB ? 1 : 0;
		});

		return filtered;
	}, [data, searchTerm, providerFilter, sortBy, sortOrder]);

	// Get unique providers for filter dropdown
	const providers = useMemo(() => {
		if (!data?.modelProviderBreakdown) return [];
		return Array.from(
			new Set(data.modelProviderBreakdown.map((item) => item.provider)),
		).map((provider) => ({
			value: provider,
			label: provider.charAt(0).toUpperCase() + provider.slice(1),
		}));
	}, [data]);

	const handleSort = (column: "savings" | "cost" | "model") => {
		if (sortBy === column) {
			setSortOrder(sortOrder === "asc" ? "desc" : "asc");
		} else {
			setSortBy(column);
			setSortOrder("desc");
		}
	};
	if (loading) {
		return (
			<div className="p-6">
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
			</div>
		);
	}

	if (
		!data ||
		!data.modelProviderBreakdown ||
		data.modelProviderBreakdown.length === 0
	) {
		return (
			<div>
				<div className="flex items-center justify-between">
					<div className="font-semibold text-xl">Model Cost Comparison</div>
					<Badge variant="secondary">
						{selectedModel
							? selectedModel.charAt(0).toUpperCase() +
								selectedModel.slice(1).replace(/-/g, " ")
							: "All Models"}
					</Badge>
				</div>
				<p className="mt-2 text-muted-foreground text-sm">
					Compare your Adaptive costs against specific models by provider
				</p>
				<div className="mt-4 flex h-64 items-center justify-center text-muted-foreground">
					No model comparison data available
				</div>
			</div>
		);
	}

	return (
		<div>
			<div className="flex items-center justify-between">
				<div className="font-semibold text-xl">Provider Cost Comparison</div>
				<Badge variant="secondary">All Providers</Badge>
			</div>
			<p className="mt-2 text-muted-foreground text-sm">
				Compare your Adaptive costs against what you would pay using each
				provider exclusively
			</p>

			{/* Filters */}
			<div className="mt-6 mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
				<div className="flex flex-1 items-center gap-4">
					<div className="relative max-w-sm flex-1">
						<Search className="-translate-y-1/2 absolute top-1/2 left-3 h-4 w-4 text-muted-foreground" />
						<Input
							placeholder="Search models or providers..."
							value={searchTerm}
							onChange={(e) => setSearchTerm(e.target.value)}
							className="pl-9"
						/>
					</div>
					<div className="flex items-center gap-2">
						<Label htmlFor="provider-filter" className="text-sm">
							Provider:
						</Label>
						<Select value={providerFilter} onValueChange={setProviderFilter}>
							<SelectTrigger className="w-40">
								<SelectValue placeholder="All" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="all">All Providers</SelectItem>
								{providers.map((provider) => (
									<SelectItem key={provider.value} value={provider.value}>
										{provider.label}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</div>
				</div>
				<div className="text-muted-foreground text-sm">
					Showing {tableData.length} results
				</div>
			</div>

			{/* Scrollable Table */}
			<div className="rounded-md border">
				<div className="max-h-96 overflow-auto">
					<Table>
						<TableHeader className="sticky top-0 bg-background">
							<TableRow>
								<TableHead>
									<button
										type="button"
										onClick={() => handleSort("model")}
										className="flex items-center gap-1 hover:text-foreground"
									>
										Model
										{sortBy === "model" &&
											(sortOrder === "asc" ? (
												<ArrowUp className="h-3 w-3" />
											) : (
												<ArrowDown className="h-3 w-3" />
											))}
									</button>
								</TableHead>
								<TableHead>Provider</TableHead>
								<TableHead className="text-right">Your Cost</TableHead>
								<TableHead className="text-right">
									<button
										type="button"
										onClick={() => handleSort("cost")}
										className="ml-auto flex items-center gap-1 hover:text-foreground"
									>
										Model Cost
										{sortBy === "cost" &&
											(sortOrder === "asc" ? (
												<ArrowUp className="h-3 w-3" />
											) : (
												<ArrowDown className="h-3 w-3" />
											))}
									</button>
								</TableHead>
								<TableHead className="text-right">Savings</TableHead>
								<TableHead className="text-right">
									<button
										type="button"
										onClick={() => handleSort("savings")}
										className="ml-auto flex items-center gap-1 hover:text-foreground"
									>
										% Saved
										{sortBy === "savings" &&
											(sortOrder === "asc" ? (
												<ArrowUp className="h-3 w-3" />
											) : (
												<ArrowDown className="h-3 w-3" />
											))}
									</button>
								</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{tableData.map((item) => {
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
											${item.adaptiveCost.toFixed(4)}
										</TableCell>
										<TableCell className="text-right font-mono">
											${item.estimatedCost.toFixed(4)}
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
													${Math.abs(item.savings).toFixed(4)}
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
			</div>

			{/* Summary Stats */}
			<div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
				<div className="rounded-lg border p-4">
					<div className="text-muted-foreground text-sm">Average Savings</div>
					<div className="font-bold text-2xl text-green-600">
						$
						{(
							tableData.reduce(
								(total, item) => total + Math.max(0, item.savings),
								0,
							) / (tableData.length || 1)
						).toFixed(4)}
					</div>
				</div>
				<div className="rounded-lg border p-4">
					<div className="text-muted-foreground text-sm">Best Savings %</div>
					<div className="font-bold text-2xl text-green-600">
						{tableData.length > 0
							? Math.max(
									...tableData.map((item) => item.savingsPercentage),
								).toFixed(1)
							: "0.0"}
						%
					</div>
				</div>
				<div className="rounded-lg border p-4">
					<div className="text-muted-foreground text-sm">Your Total Spend</div>
					<div className="font-bold text-2xl">
						${data.totalSpend.toFixed(4)}
					</div>
				</div>
			</div>
		</div>
	);
}
