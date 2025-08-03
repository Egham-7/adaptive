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
import type { DashboardData } from "@/types/api-platform/dashboard";

interface ProviderComparisonTableProps {
	data: DashboardData | null;
	loading: boolean;
}

export function ProviderComparisonTable({
	data,
	loading,
}: ProviderComparisonTableProps) {
	if (loading) {
		return (
			<Card>
				<CardContent className="p-6">
					<div className="animate-pulse">
						<div className="mb-4 h-4 w-1/3 rounded bg-muted" />
						<div className="space-y-3">
							{[...Array(5)].map((_, i) => (
								/* biome-ignore lint/suspicious/noArrayIndexKey: skeleton loading doesn't need stable keys */
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

	if (!data || !data.providers || data.providers.length === 0) {
		return (
			<Card>
				<CardHeader>
					<CardTitle>Provider Cost Comparison</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex h-64 items-center justify-center text-muted-foreground">
						No provider data available
					</div>
				</CardContent>
			</Card>
		);
	}

	const sortedProviders = [...data.providers].sort(
		(a, b) => b.comparisonCosts.single - a.comparisonCosts.single,
	);

	const formatCurrency = (amount: number) => {
		return new Intl.NumberFormat("en-US", {
			style: "currency",
			currency: "USD",
			minimumFractionDigits: 2,
		}).format(amount);
	};

	const calculateSavings = (adaptive: number, single: number) => {
		const savings = single - adaptive;
		const percentage = single > 0 ? (savings / single) * 100 : 0;
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
								<TableHead>Provider</TableHead>
								<TableHead className="text-right">Your Cost</TableHead>
								<TableHead className="text-right">
									Single Provider Cost
								</TableHead>
								<TableHead className="text-right">Savings</TableHead>
								<TableHead className="text-right">% Saved</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{sortedProviders.map((provider) => {
								const savings = calculateSavings(
									data.totalSpend,
									provider.comparisonCosts.single,
								);
								const isPositiveSavings = savings.amount > 0;

								return (
									<TableRow key={provider.id}>
										<TableCell>
											<div className="flex items-center gap-3">
												<Image
													width={24}
													height={24}
													src={provider.icon}
													alt={provider.name}
													className="rounded"
												/>
												<span className="font-medium">{provider.name}</span>
											</div>
										</TableCell>
										<TableCell className="text-right font-mono">
											{formatCurrency(data.totalSpend)}
										</TableCell>
										<TableCell className="text-right font-mono">
											{formatCurrency(provider.comparisonCosts.single)}
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
													{formatCurrency(Math.abs(savings.amount))}
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
												{savings.percentage.toFixed(1)}%
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
						<div className="text-muted-foreground text-sm">Total Saved</div>
						<div className="font-bold text-2xl text-green-600">
							{formatCurrency(data.totalSavings)}
						</div>
					</div>
					<div className="rounded-lg border p-4">
						<div className="text-muted-foreground text-sm">Average Savings</div>
						<div className="font-bold text-2xl text-green-600">
							{data.savingsPercentage.toFixed(1)}%
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
