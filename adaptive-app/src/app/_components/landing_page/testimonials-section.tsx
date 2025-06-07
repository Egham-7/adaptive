import Image from "next/image";

export default function TestimonialsSection() {
	return (
		<section className="py-16 md:py-32">
			<div className="mx-auto max-w-5xl px-6">
				<div className="mx-auto max-w-2xl">
					<blockquote>
						<p className="font-semibold text-lg sm:text-xl md:text-3xl">
							Using Adaptive has been like unlocking a secret productivity
							superpower. It&apos;s the perfect fusion of simplicity and
							versatility, enabling us to create workflows that are as efficient
							as they are user-friendly.
						</p>

						<div className="mt-12 flex items-center gap-6">
							<Image
								className="h-7 w-fit dark:invert"
								src="https://html.tailus.io/blocks/customers/nvidia.svg"
								alt="Nvidia Logo"
								height={28}
								width={100} // Approximate width, adjust as needed
							/>

							<div className="space-y-1 border-l pl-6">
								<cite className="font-medium">Sarah Johnson</cite>
								<span className="block text-muted-foreground text-sm">
									CTO, Nvidia
								</span>
							</div>
						</div>
					</blockquote>
				</div>
			</div>
		</section>
	);
}
