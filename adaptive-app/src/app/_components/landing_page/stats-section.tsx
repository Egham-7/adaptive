export default function StatsSection() {
	return (
		<section className="py-12 md:py-20">
			<div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
				<div className="relative z-10 mx-auto max-w-xl space-y-6 text-center">
					<h2 className="font-medium text-4xl lg:text-5xl">
						Adaptive in numbers
					</h2>
					<p className="text-muted-foreground">
						Adaptive is evolving to be more than just a platform. It supports an
						entire ecosystem of tools and services helping developers and
						businesses adapt to changing needs.
					</p>
				</div>

				<div className="grid gap-12 divide-y *:text-center md:grid-cols-3 md:gap-2 md:divide-x md:divide-y-0">
					<div className="space-y-4 pt-12 md:pt-0">
						<div className="font-bold text-5xl text-primary">+1200</div>
						<p className="text-muted-foreground">Stars on GitHub</p>
					</div>
					<div className="space-y-4 pt-12 md:pt-0">
						<div className="font-bold text-5xl text-primary">22 Million</div>
						<p className="text-muted-foreground">Active Users</p>
					</div>
					<div className="space-y-4 pt-12 md:pt-0">
						<div className="font-bold text-5xl text-primary">+500</div>
						<p className="text-muted-foreground">Powered Apps</p>
					</div>
				</div>
			</div>
		</section>
	);
}
