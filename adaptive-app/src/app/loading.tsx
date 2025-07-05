export default function Loading() {
	return (
		<div className="min-h-screen bg-background">
			{/* Header Skeleton */}
			<div className="border-border border-b">
				<div className="container mx-auto px-4 py-4">
					<div className="flex items-center justify-between">
						<div className="h-8 w-32 animate-pulse rounded bg-muted" />
						<div className="flex items-center gap-4">
							<div className="h-8 w-16 animate-pulse rounded bg-muted" />
							<div className="h-8 w-20 animate-pulse rounded bg-muted" />
							<div className="h-8 w-24 animate-pulse rounded bg-muted" />
						</div>
					</div>
				</div>
			</div>

			{/* Hero Section Skeleton */}
			<div className="container mx-auto px-4 py-20">
				<div className="space-y-6 text-center">
					<div className="space-y-4">
						<div className="mx-auto h-12 w-3/4 animate-pulse rounded bg-muted" />
						<div className="mx-auto h-12 w-1/2 animate-pulse rounded bg-muted" />
					</div>
					<div className="mx-auto h-6 w-2/3 animate-pulse rounded bg-muted" />
					<div className="flex items-center justify-center gap-4">
						<div className="h-12 w-32 animate-pulse rounded bg-muted" />
						<div className="h-12 w-28 animate-pulse rounded bg-muted" />
					</div>
				</div>
			</div>

			{/* Charts Section Skeleton */}
			<div className="container mx-auto px-4 py-16">
				<div className="space-y-8">
					<div className="space-y-4 text-center">
						<div className="mx-auto h-8 w-48 animate-pulse rounded bg-muted" />
						<div className="mx-auto h-4 w-96 animate-pulse rounded bg-muted" />
					</div>
					<div className="grid gap-8 lg:grid-cols-2">
						{Array.from({ length: 2 }).map((_, i) => (
							<div
								/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
								key={`chart-${i}`}
								className="rounded-lg border border-border bg-card p-6"
							>
								<div className="space-y-4">
									<div className="h-6 w-32 animate-pulse rounded bg-muted" />
									<div className="h-64 w-full animate-pulse rounded bg-muted" />
								</div>
							</div>
						))}
					</div>
				</div>
			</div>

			{/* Features Section Skeleton */}
			<div className="container mx-auto px-4 py-16">
				<div className="space-y-12">
					<div className="space-y-4 text-center">
						<div className="mx-auto h-8 w-40 animate-pulse rounded bg-muted" />
						<div className="mx-auto h-4 w-80 animate-pulse rounded bg-muted" />
					</div>
					<div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
						{Array.from({ length: 6 }).map((_, i) => (
							<div
								/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
								key={`feature-${i}`}
								className="rounded-lg border border-border bg-card p-6"
							>
								<div className="space-y-4">
									<div className="h-12 w-12 animate-pulse rounded bg-muted" />
									<div className="h-6 w-32 animate-pulse rounded bg-muted" />
									<div className="space-y-2">
										<div className="h-4 w-full animate-pulse rounded bg-muted" />
										<div className="h-4 w-3/4 animate-pulse rounded bg-muted" />
									</div>
								</div>
							</div>
						))}
					</div>
				</div>
			</div>

			{/* Content Section Skeleton */}
			<div className="container mx-auto px-4 py-16">
				<div className="grid items-center gap-12 lg:grid-cols-2">
					<div className="space-y-6">
						<div className="h-8 w-64 animate-pulse rounded bg-muted" />
						<div className="space-y-3">
							<div className="h-4 w-full animate-pulse rounded bg-muted" />
							<div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
							<div className="h-4 w-4/5 animate-pulse rounded bg-muted" />
						</div>
						<div className="h-12 w-32 animate-pulse rounded bg-muted" />
					</div>
					<div className="h-80 w-full animate-pulse rounded bg-muted" />
				</div>
			</div>

			{/* Pricing Section Skeleton */}
			<div className="container mx-auto px-4 py-16">
				<div className="space-y-12">
					<div className="space-y-4 text-center">
						<div className="mx-auto h-8 w-32 animate-pulse rounded bg-muted" />
						<div className="mx-auto h-4 w-64 animate-pulse rounded bg-muted" />
					</div>
					<div className="grid gap-8 md:grid-cols-3">
						{Array.from({ length: 3 }).map((_, i) => (
							<div
								/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
								key={`pricing-${i}`}
								className="rounded-lg border border-border bg-card p-6"
							>
								<div className="space-y-6">
									<div className="space-y-2">
										<div className="h-6 w-24 animate-pulse rounded bg-muted" />
										<div className="h-8 w-32 animate-pulse rounded bg-muted" />
									</div>
									<div className="space-y-3">
										{Array.from({ length: 5 }).map((_, j) => (
											<div
												/* biome-ignore lint/suspicious/noArrayIndexKey: Static loading skeleton */
												key={`pricing-feature-${i}-${j}`}
												className="h-4 w-full animate-pulse rounded bg-muted"
											/>
										))}
									</div>
									<div className="h-12 w-full animate-pulse rounded bg-muted" />
								</div>
							</div>
						))}
					</div>
				</div>
			</div>

			{/* CTA Section Skeleton */}
			<div className="container mx-auto px-4 py-16">
				<div className="space-y-6 rounded-lg border border-border bg-card p-12 text-center">
					<div className="mx-auto h-8 w-80 animate-pulse rounded bg-muted" />
					<div className="mx-auto h-4 w-96 animate-pulse rounded bg-muted" />
					<div className="mx-auto h-12 w-36 animate-pulse rounded bg-muted" />
				</div>
			</div>
		</div>
	);
}
