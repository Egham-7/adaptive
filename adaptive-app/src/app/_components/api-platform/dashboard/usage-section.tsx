"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer } from "@/components/ui/chart";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";
import { RequestsChart } from "./charts/request-chart";
import { TokenChart } from "./charts/token-chart";
import { UsageChart } from "./charts/usage-chart";
import { ChartSkeleton } from "./loading-skeleton";

interface UsageSectionProps {
	data: DashboardData | null;
	loading: boolean;
	selectedProvider: string;
	providers: Provider[];
}

export function UsageSection({
	data,
	loading,
	selectedProvider,
	providers,
}: UsageSectionProps) {
	if (loading) {
		return (
			<div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
				<div className="xl:col-span-3">
					<ChartSkeleton />
				</div>
				<div className="space-y-4">
					<ChartSkeleton height="120px" />
					<ChartSkeleton height="120px" />
					<ChartSkeleton height="120px" />
				</div>
			</div>
		);
	}

	if (!data) return null;

	const currentProvider = providers.find((p) => p.id === selectedProvider);
	const totalSpend = data.totalSpend;
	const totalComparison = currentProvider?.comparisonCosts.single || 0;
	const totalSavings = totalComparison - totalSpend;
	const savingsPercentage = ((totalSavings / totalComparison) * 100).toFixed(1);

	return (
		<div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
			{/* Main Usage Chart */}
			<div className="xl:col-span-3">
				<Card>
					<CardHeader>
						<div className="flex items-center justify-between">
							<div>
								<CardTitle className="mb-1">Total Spend</CardTitle>
								<div className="flex items-baseline gap-4">
									<span className="font-bold text-3xl text-foreground">
										${totalSpend.toFixed(2)}
									</span>
									<span className="text-muted-foreground text-sm">
										vs ${totalComparison.toFixed(2)} ({currentProvider?.name})
									</span>
								</div>
								<div className="mt-2 flex items-center gap-2">
									<span className="font-medium text-green-600 text-sm dark:text-green-400">
										${totalSavings.toFixed(2)} saved ({savingsPercentage}%)
									</span>
								</div>
							</div>
							<div className="text-right">
								<Select defaultValue="1d">
									<SelectTrigger className="w-20">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="1d">1d</SelectItem>
										<SelectItem value="7d">7d</SelectItem>
										<SelectItem value="30d">30d</SelectItem>
									</SelectContent>
								</Select>
							</div>
						</div>
					</CardHeader>
					<CardContent>
						<UsageChart
							data={data.usageData}
							providerName={currentProvider?.name}
						/>
					</CardContent>
				</Card>
			</div>

			{/* Right Sidebar Stats */}
			<div className="space-y-4">
				{/* Budget Card */}
				<Card>
					<CardHeader className="pb-2">
						<CardTitle className="font-medium text-sm">
							Monthly budget
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="mb-3 font-semibold text-lg">
							${totalSpend.toFixed(2)} / $50.00
						</div>

						<ChartContainer
							config={{
								used: {
									label: "Used",
									color: "#3b82f6",
								},
								remaining: {
									label: "Remaining",
									color: "#e5e7eb",
								},
							}}
						>
							<ResponsiveContainer width="100%" height="100%">
								<PieChart>
									<Pie
										data={[
											{ name: "used", value: totalSpend, fill: "#3b82f6" },
											{
												name: "remaining",
												value: Math.max(0, 50 - totalSpend),
												fill: "#e5e7eb",
											},
										]}
										cx="50%"
										cy="50%"
										innerRadius={20}
										outerRadius={32}
										startAngle={90}
										endAngle={450}
										dataKey="value"
									>
										{[
											{ name: "used", value: totalSpend, fill: "#3b82f6" },
											{
												name: "remaining",
												value: Math.max(0, 50 - totalSpend),
												fill: "#e5e7eb",
											},
										].map((entry, _index) => (
											<Cell key={entry.name} fill={entry.fill} />
										))}
									</Pie>
								</PieChart>
							</ResponsiveContainer>
						</ChartContainer>

						<div className="text-muted-foreground text-xs">
							Resets in 7 days.{" "}
							<Button variant="link" size="sm" className="h-auto p-0 text-xs">
								Edit budget
							</Button>
						</div>
					</CardContent>
				</Card>

				{/* Total Tokens */}
				<Card>
					<CardHeader className="pb-2">
						<CardTitle className="font-medium text-sm">Total tokens</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="mb-3 font-semibold text-lg">
							{data.totalTokens.toLocaleString()}
						</div>
						<TokenChart data={data.tokenData} />
					</CardContent>
				</Card>

				{/* Total Requests */}
				<Card>
					<CardHeader className="pb-2">
						<CardTitle className="font-medium text-sm">
							Total requests
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="mb-3 font-semibold text-lg">
							{data.totalRequests.toLocaleString()}
						</div>
						<RequestsChart data={data.requestData} />
					</CardContent>
				</Card>
			</div>
		</div>
	);
}
