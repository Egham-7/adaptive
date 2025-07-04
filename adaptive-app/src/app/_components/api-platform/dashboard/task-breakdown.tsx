"use client";

import { Code, FileText, Globe, HelpCircle, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardData, Provider } from "@/types/api-platform/dashboard";
import { TaskBreakdownSkeleton } from "./loading-skeleton";

interface TaskBreakdownProps {
	data: DashboardData | null;
	loading: boolean;
	selectedProvider: string;
	providers: Provider[];
}

const getTaskIcon = (taskName: string) => {
	switch (taskName) {
		case "Code Generation":
			return <Code className="h-5 w-5 text-blue-500" />;
		case "Open Q&A":
			return <HelpCircle className="h-5 w-5 text-green-500" />;
		case "Summarization":
			return <FileText className="h-5 w-5 text-orange-500" />;
		case "Translation":
			return <Globe className="h-5 w-5 text-purple-500" />;
		default:
			return <TrendingUp className="h-5 w-5 text-gray-500" />;
	}
};

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
		<Card>
			<CardHeader>
				<div className="flex items-center justify-between">
					<CardTitle>Task Type Performance</CardTitle>
					<Button variant="ghost" size="sm">
						View detailed breakdown
					</Button>
				</div>
			</CardHeader>
			<CardContent>
				<div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
					{data.taskBreakdown.map((task, _index) => (
						<Card key={task.name} className="border-border">
							<CardContent className="p-4">
								<div className="mb-3 flex items-center justify-between">
									<div className="flex items-center gap-3">
										{getTaskIcon(task.name)}
										<div>
											<h3 className="font-medium text-foreground">
												{task.name}
											</h3>
											<div className="mt-1 flex items-center gap-2 text-muted-foreground text-xs">
												<span>{task.requests} requests</span>
												<span>•</span>
												<span>{task.inputTokens} in</span>
												<span>•</span>
												<span>{task.outputTokens} out</span>
											</div>
										</div>
									</div>
									<div className="text-right">
										<div className="font-semibold text-foreground text-lg">
											{task.cost}
										</div>
										<Badge variant="secondary" className="text-xs">
											{task.percentage}% saved
										</Badge>
									</div>
								</div>

								{/* Simple progress bar */}
								<div className="space-y-2">
									<div className="flex justify-between text-muted-foreground text-xs">
										<span>Adaptive: {task.cost}</span>
										<span>
											{currentProvider?.name || "Single provider"}:{" "}
											{task.comparisonCost}
										</span>
									</div>
									<div className="h-2 overflow-hidden rounded-full bg-muted">
										<div
											className="h-full rounded-full bg-primary transition-all duration-300"
											style={{ width: `${100 - task.percentage}%` }}
										/>
									</div>
								</div>
							</CardContent>
						</Card>
					))}
				</div>
			</CardContent>
		</Card>
	);
}
