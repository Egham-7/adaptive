export default function Loading() {
	return (
		<div className="min-h-screen bg-background p-6">
			<div className="mx-auto max-w-6xl space-y-6">
				{/* Header Skeleton */}
				<div className="mb-8 flex items-center gap-3">
					<div className="h-8 w-8 animate-pulse rounded bg-muted" />
					<div className="h-8 w-24 animate-pulse rounded bg-muted" />
				</div>

				{/* Tabs Skeleton */}
				<div className="space-y-6">
					<div className="grid h-10 w-full animate-pulse grid-cols-3 rounded-lg bg-muted" />

					{/* Tab Content Skeleton */}
					<div className="space-y-6">
						{/* Providers Tab Content */}
						<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
							{Array.from({ length: 6 }).map((_, i) => (
								<div
									/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
									key={`provider-${i}`}
									className="rounded-lg border border-border bg-card p-6"
								>
									<div className="space-y-4">
										{/* Provider header */}
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-3">
												<div className="h-10 w-10 animate-pulse rounded bg-muted" />
												<div className="space-y-1">
													<div className="h-5 w-20 animate-pulse rounded bg-muted" />
													<div className="h-3 w-16 animate-pulse rounded bg-muted" />
												</div>
											</div>
											<div className="h-6 w-12 animate-pulse rounded-full bg-muted" />
										</div>

										{/* Provider settings */}
										<div className="space-y-3">
											<div className="space-y-2">
												<div className="h-4 w-16 animate-pulse rounded bg-muted" />
												<div className="h-10 w-full animate-pulse rounded bg-muted" />
											</div>
											<div className="space-y-2">
												<div className="h-4 w-20 animate-pulse rounded bg-muted" />
												<div className="h-10 w-full animate-pulse rounded bg-muted" />
											</div>
											<div className="space-y-2">
												<div className="h-4 w-24 animate-pulse rounded bg-muted" />
												<div className="h-20 w-full animate-pulse rounded bg-muted" />
											</div>
										</div>

										{/* Action buttons */}
										<div className="flex justify-end gap-2">
											<div className="h-9 w-16 animate-pulse rounded bg-muted" />
											<div className="h-9 w-12 animate-pulse rounded bg-muted" />
										</div>
									</div>
								</div>
							))}
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}
