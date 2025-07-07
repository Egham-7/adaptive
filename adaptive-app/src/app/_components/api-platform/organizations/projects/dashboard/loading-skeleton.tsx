"use client";

import { Skeleton } from "@/components/ui/skeleton";

export function MetricCardSkeleton() {
	return (
		<div className="rounded-xl border border-gray-200 bg-white p-6 dark:border-[#1F1F23] dark:bg-[#0F0F12]">
			<div className="mb-4 flex items-center justify-between">
				<Skeleton className="h-10 w-10 rounded-lg" />
				<Skeleton className="h-6 w-16" />
			</div>
			<div className="space-y-2">
				<Skeleton className="h-4 w-24" />
				<Skeleton className="h-8 w-20" />
				<Skeleton className="h-3 w-32" />
			</div>
		</div>
	);
}

export function ChartSkeleton({ height = "300px" }: { height?: string }) {
	return (
		<div className="rounded-xl border border-gray-200 bg-white p-6 dark:border-[#1F1F23] dark:bg-[#0F0F12]">
			<div className="mb-6 flex items-center justify-between">
				<div className="space-y-2">
					<Skeleton className="h-6 w-32" />
					<Skeleton className="h-4 w-48" />
				</div>
				<Skeleton className="h-8 w-20" />
			</div>
			<Skeleton className="w-full" style={{ height }} />
		</div>
	);
}

export function TaskBreakdownSkeleton() {
	return (
		<div className="rounded-xl border border-gray-200 bg-white p-6 dark:border-[#1F1F23] dark:bg-[#0F0F12]">
			<div className="mb-6 flex items-center justify-between">
				<Skeleton className="h-6 w-48" />
				<Skeleton className="h-4 w-32" />
			</div>
			<div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
				{Array.from({ length: 4 }).map((_, i) => (
					<div
						// biome-ignore lint/suspicious/noArrayIndexKey: Using index for skeleton components is acceptable
						key={`skeleton-${i}`}
						className="rounded-lg border border-gray-200 p-4 dark:border-[#1F1F23]"
					>
						<div className="mb-4 flex items-center justify-between">
							<div className="flex items-center gap-3">
								<Skeleton className="h-6 w-6" />
								<div className="space-y-1">
									<Skeleton className="h-5 w-24" />
									<Skeleton className="h-4 w-32" />
								</div>
							</div>
							<div className="space-y-1 text-right">
								<Skeleton className="h-6 w-16" />
								<Skeleton className="h-4 w-20" />
							</div>
						</div>
						<div className="space-y-2">
							<div className="flex justify-between">
								<Skeleton className="h-4 w-20" />
								<Skeleton className="h-4 w-12" />
							</div>
							<div className="flex justify-between">
								<Skeleton className="h-4 w-24" />
								<Skeleton className="h-4 w-16" />
							</div>
							<Skeleton className="h-2 w-full rounded-full" />
						</div>
					</div>
				))}
			</div>
		</div>
	);
}
