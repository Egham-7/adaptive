export default function Loading() {
	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div className="h-8 w-24 animate-pulse rounded bg-muted" />
				<div className="h-10 w-48 animate-pulse rounded bg-muted" />
			</div>

			{/* Description */}
			<div className="space-y-3">
				<div className="h-4 w-full animate-pulse rounded bg-muted" />
				<div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
				<div className="h-4 w-4/5 animate-pulse rounded bg-muted" />
			</div>

			{/* Table */}
			<div className="overflow-hidden rounded-lg border border-border bg-card">
				<div className="overflow-x-auto">
					<table className="w-full">
						<thead className="bg-muted/50">
							<tr>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-12 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-20 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-16 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-20 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-24 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-left">
									<div className="h-4 w-24 animate-pulse rounded bg-muted" />
								</th>
								<th className="px-6 py-3 text-right">
									<div className="ml-auto h-4 w-16 animate-pulse rounded bg-muted" />
								</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-border">
							{Array.from({ length: 5 }).map((_, i) => (
								<tr key={`api-key-row-${i}`} className="hover:bg-muted/50">
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-24 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-20 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-20 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-16 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-20 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4">
										<div className="h-4 w-8 animate-pulse rounded bg-muted" />
									</td>
									<td className="whitespace-nowrap px-6 py-4 text-right">
										<div className="flex items-center justify-end gap-2">
											<div className="h-8 w-8 animate-pulse rounded bg-muted" />
											<div className="h-8 w-8 animate-pulse rounded bg-muted" />
										</div>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</div>
		</div>
	);
}
