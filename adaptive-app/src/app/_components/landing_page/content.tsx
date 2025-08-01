import { Cpu, Zap } from "lucide-react";
import { LogoCarousel } from "./logo-carousel";

const logos1 = [
	{ name: "OpenAI", id: 1, img: "/logos/openai.webp" },
	{ name: "Gemini", id: 2, img: "/logos/google.svg" },
	{ name: "DeepSeek", id: 3, img: "/logos/deepseek.svg" },
	{ name: "Anthropic", id: 4, img: "/logos/anthropic.jpeg" },
];

const logos2 = [
	{ name: "Grok", id: 6, img: "/logos/grok.svg" },
	{ name: "Mistral", id: 8, img: "/logos/mistral.png" },
	{ name: "Cohere", id: 9, img: "/logos/cohere.png" },
	{ name: "Hugging Face", id: 10, img: "/logos/huggingface.png" },
];

export default function ContentSection() {
	return (
		<section id="solution" className="py-16 md:py-32">
			<div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
				<h2 className="relative z-10 max-w-xl font-medium text-4xl lg:text-5xl">
					The Adaptive platform optimizes AI performance across all providers.
				</h2>

				<div className="grid gap-6 md:grid-cols-2 md:gap-12 lg:gap-24">
					<div className="relative space-y-4">
						<p className="text-muted-foreground">
							Adaptive is more than just model routing.{" "}
							<span className="font-bold text-accent-foreground">
								It's an intelligent inference platform
							</span>{" "}
							— with smart protocols that handle your entire request flow
							automatically.
						</p>

						<p className="text-muted-foreground">
							From intuitive interfaces to powerful APIs, we help developers and
							businesses get better AI results while reducing costs and ensuring
							uptime.
						</p>

						<div className="grid grid-cols-2 gap-3 pt-6 md:gap-4">
							<div className="space-y-3">
								<div className="flex items-center gap-2">
									<Zap className="size-4" />
									<h3 className="font-medium text-sm">Lightning Fast</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Smart caching and optimization deliver lightning-fast
									responses.
								</p>
							</div>
							<div className="space-y-2">
								<div className="flex items-center gap-2">
									<Cpu className="size-4" />
									<h3 className="font-medium text-sm">Powerful</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Advanced algorithms automatically optimize AI performance.
								</p>
							</div>
						</div>
					</div>

					<div className="relative mt-6 md:mt-0">
						<div className="rounded-2xl bg-[linear-gradient(to_bottom,var(--color-input),transparent)] p-px dark:bg-[linear-gradient(to_bottom,var(--color-card),transparent)]">
							<div className="p-6 md:p-8">
								<LogoCarousel logos={logos1} columnCount={3} />
							</div>
						</div>
						<div className="p-6 md:p-8">
							<LogoCarousel logos={logos2} columnCount={3} />
						</div>
					</div>
				</div>
			</div>
		</section>
	);
}
