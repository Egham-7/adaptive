export default function Loading() {
	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto max-w-6xl px-4 py-8">
				{/* Header Skeleton */}
				<div className="mb-8">
					<div className="mb-4">
						<div className="h-10 w-48 animate-pulse rounded bg-muted" />
					</div>
					<div className="mb-2 h-8 w-64 animate-pulse rounded bg-muted" />
					<div className="mb-6 h-4 w-96 animate-pulse rounded bg-muted" />

					{/* Stats */}
					<div className="flex flex-wrap gap-8">
						{Array.from({ length: 2 }).map((_, i) => (
							/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
							<div key={`stat-${i}`} className="flex items-center gap-2">
								<div className="h-10 w-10 animate-pulse rounded-lg bg-muted" />
								<div className="space-y-1">
									<div className="h-5 w-8 animate-pulse rounded bg-muted" />
									<div className="h-4 w-16 animate-pulse rounded bg-muted" />
								</div>
							</div>
						))}
					</div>
				</div>

				{/* Search Skeleton */}
				<div className="mb-8">
					<div className="h-12 w-80 animate-pulse rounded bg-muted" />
				</div>

				{/* Projects Grid Skeleton */}
				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
					{Array.from({ length: 6 }).map((_, i) => (
						<div
							/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
							key={`project-${i}`}
							className="rounded-lg border border-border bg-card p-6"
						>
							<div className="space-y-4">
								{/* Header with icon and title */}
								<div className="mb-3 flex items-start justify-between">
									<div className="flex items-center gap-3">
										<div className="h-10 w-10 animate-pulse rounded-lg bg-muted" />
										<div className="h-6 w-32 animate-pulse rounded bg-muted" />
									</div>
									<div className="h-4 w-4 animate-pulse rounded bg-muted" />
								</div>

								{/* Status and badge */}
								<div className="mb-3 flex items-center gap-2">
									<div className="h-4 w-4 animate-pulse rounded bg-muted" />
									<div className="h-6 w-16 animate-pulse rounded bg-muted" />
								</div>

								{/* Progress bar */}
								<div className="mb-3">
									<div className="mb-1 flex justify-between text-sm">
										<div className="h-4 w-16 animate-pulse rounded bg-muted" />
										<div className="h-4 w-8 animate-pulse rounded bg-muted" />
									</div>
									<div className="h-2 w-full animate-pulse rounded bg-muted" />
								</div>

								{/* Description */}
								<div className="mb-4 space-y-2">
									<div className="h-4 w-full animate-pulse rounded bg-muted" />
									<div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
								</div>

								{/* Stats */}
								<div className="space-y-3">
									<div className="flex items-center justify-between">
										<div className="h-4 w-20 animate-pulse rounded bg-muted" />
										<div className="h-4 w-24 animate-pulse rounded bg-muted" />
									</div>
									<div className="h-4 w-28 animate-pulse rounded bg-muted" />
								</div>
							</div>
						</div>
					))}
				</div>
			</div>
		</div>
	);
}
