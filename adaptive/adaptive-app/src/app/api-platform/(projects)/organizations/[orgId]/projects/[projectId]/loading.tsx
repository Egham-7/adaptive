export default function Loading() {
	return (
		<div className="space-y-8">
			{/* Header Section Skeleton */}
			<div className="flex items-center justify-between">
				<div className="h-8 w-48 animate-pulse rounded bg-muted" />
				<div className="flex items-center gap-3">
					<div className="h-10 w-56 animate-pulse rounded bg-muted" />
					<div className="h-10 w-48 animate-pulse rounded bg-muted" />
					<div className="h-10 w-20 animate-pulse rounded bg-muted" />
					<div className="h-10 w-20 animate-pulse rounded bg-muted" />
				</div>
			</div>

			{/* Key Metrics Section Skeleton */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<div className="h-6 w-48 animate-pulse rounded bg-muted" />
					<div className="h-4 w-32 animate-pulse rounded bg-muted" />
				</div>

				{/* Metrics Cards */}
				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
					{Array.from({ length: 4 }).map((_, i) => (
						<div
							key={`metric-${i}`}
							className="rounded-xl border border-border bg-card p-6"
						>
							<div className="mb-4 flex items-center justify-between">
								<div className="h-10 w-10 animate-pulse rounded-lg bg-muted" />
								<div className="flex items-center gap-1">
									<div className="h-4 w-4 animate-pulse rounded bg-muted" />
									<div className="h-4 w-12 animate-pulse rounded bg-muted" />
								</div>
							</div>
							<div className="space-y-1">
								<div className="h-4 w-20 animate-pulse rounded bg-muted" />
								<div className="h-8 w-24 animate-pulse rounded bg-muted" />
								<div className="h-3 w-32 animate-pulse rounded bg-muted" />
							</div>
						</div>
					))}
				</div>
			</section>

			{/* Divider */}
			<div className="border-border border-t" />

			{/* Usage Analytics Section Skeleton */}
			<section className="space-y-4">
				<div className="flex items-center justify-between">
					<div className="h-6 w-36 animate-pulse rounded bg-muted" />
					<div className="h-4 w-40 animate-pulse rounded bg-muted" />
				</div>

				<div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
					{/* Main Chart Area */}
					<div className="lg:col-span-2">
						<div className="rounded-lg border border-border bg-card p-6">
							<div className="space-y-4">
								<div className="h-6 w-32 animate-pulse rounded bg-muted" />
								<div className="h-80 w-full animate-pulse rounded bg-muted" />
							</div>
						</div>
					</div>

					{/* Side Chart */}
					<div className="rounded-lg border border-border bg-card p-6">
						<div className="space-y-4">
							<div className="flex items-center justify-between">
								<div className="h-6 w-32 animate-pulse rounded bg-muted" />
								<div className="h-6 w-20 animate-pulse rounded bg-muted" />
							</div>
							<div className="h-64 w-full animate-pulse rounded bg-muted" />
						</div>
					</div>
				</div>
			</section>

			{/* Divider */}
			<div className="border-border border-t" />
		</div>
	);
}
