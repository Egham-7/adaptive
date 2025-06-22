import ComparisonChart from "./ComparisonChart";

export default function ChartsSection() {
	

	return (
		<section className="overflow-hidden py-16 md:py-32">
			<div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-12">
				<div className="relative z-10 max-w-2xl">
					<h2 className="font-semibold text-4xl lg:text-5xl">
						Best Price for the Best Qualtiy
					</h2>
					
					<div><p className="mt-6 text-lg">
						We automatically choose the right model based on any given task.
					</p></div>
					<div><p className="text-lg">
						Optimizing both cost and quality.
					</p></div>
				</div>
				<ComparisonChart />

			</div>
		</section>
	);
}
