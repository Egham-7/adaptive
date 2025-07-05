export default function Loading() {
	return (
		<div className="flex min-h-screen flex-col items-center justify-center bg-background p-6">
			<div className="mx-auto flex w-full max-w-3xl flex-col items-center">
				{/* Welcome screen skeleton */}
				<div className="mb-8 text-center">
					<div className="mx-auto mb-4 h-8 w-64 animate-pulse rounded-lg bg-muted" />
					<div className="mx-auto h-5 w-48 animate-pulse rounded bg-muted" />
				</div>

				{/* Prompt suggestions skeleton */}
				<div className="mb-8 w-full">
					<div className="mb-4 h-6 w-40 animate-pulse rounded bg-muted" />
					<div className="grid grid-cols-1 gap-3 md:grid-cols-2">
						{[1, 2, 3, 4].map((i) => (
							<div key={i} className="h-12 animate-pulse rounded-lg bg-muted" />
						))}
					</div>
				</div>

				{/* Input area skeleton */}
				<div className="mb-6 w-full">
					<div className="relative">
						<div className="h-14 w-full animate-pulse rounded-lg bg-input" />
						<div className="absolute top-3 right-3 h-8 w-8 animate-pulse rounded-full bg-muted" />
					</div>
				</div>

				{/* Status bar skeleton */}
				<div className="flex items-center gap-4 text-sm">
					<div className="h-4 w-32 animate-pulse rounded bg-muted" />
					<div className="h-4 w-24 animate-pulse rounded bg-muted" />
				</div>
			</div>
		</div>
	);
}
