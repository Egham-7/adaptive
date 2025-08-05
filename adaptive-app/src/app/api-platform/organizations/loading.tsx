export default function Loading() {
	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto max-w-6xl px-4 py-8">
				{/* Header Skeleton */}
				<div className="mb-8">
					<div className="mb-2 h-8 w-48 animate-pulse rounded bg-muted" />
					<div className="h-4 w-96 animate-pulse rounded bg-muted" />
				</div>

				{/* Search Skeleton */}
				<div className="mb-8">
					<div className="h-12 w-80 animate-pulse rounded bg-muted" />
				</div>

				{/* Organizations Grid Skeleton */}
				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
					{Array.from({ length: 4 }).map((_, i) => (
						<div
							key={`org-${i}`}
							className="rounded-lg border border-border bg-card p-6"
						>
							<div className="space-y-4">
								{/* Header with icon and title */}
								<div className="flex items-start justify-between">
									<div className="flex items-center gap-3">
										<div className="h-10 w-10 animate-pulse rounded-lg bg-muted" />
										<div className="space-y-2">
											<div className="h-6 w-32 animate-pulse rounded bg-muted" />
											<div className="h-4 w-16 animate-pulse rounded bg-muted" />
										</div>
									</div>
									<div className="h-5 w-5 animate-pulse rounded bg-muted" />
								</div>

								{/* Description */}
								<div className="space-y-2">
									<div className="h-4 w-full animate-pulse rounded bg-muted" />
									<div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
								</div>

								{/* Stats */}
								<div className="flex items-center gap-4">
									<div className="flex items-center gap-1">
										<div className="h-4 w-4 animate-pulse rounded bg-muted" />
										<div className="h-4 w-20 animate-pulse rounded bg-muted" />
									</div>
									<div className="h-4 w-16 animate-pulse rounded bg-muted" />
								</div>
							</div>
						</div>
					))}
				</div>
			</div>
		</div>
	);
}
