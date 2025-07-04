import { Cpu, Zap } from "lucide-react";
import { LogoCarousel } from "./logo-carousel";

const logos1 = [
	{ name: "OpenAI", id: 1, img: "/logos/openai.webp" },
	{ name: "Gemini", id: 2, img: "/logos/google.svg" },
	{ name: "DeepSeek", id: 3, img: "/logos/deepseek.svg" },
	{ name: "Llama", id: 2, img: "/logos/meta.png" },
];

const logos2 = [
	{ name: "Anthropic", id: 1, img: "/logos/anthropic.jpeg" },
	{ name: "Llama", id: 2, img: "/logos/meta.png" },
	{ name: "Grok", id: 3, img: "/logos/grok.svg" },
	{ name: "OpenAI", id: 4, img: "/logos/openai.webp" },
];

export default function ContentSection() {
	return (
		<section id="solution" className="py-16 md:py-32">
			<div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
				<h2 className="relative z-10 max-w-xl font-medium text-4xl lg:text-5xl">
					The Adaptive ecosystem brings together our powerful models.
				</h2>

				<div className="grid gap-6 sm:grid-cols-2 md:gap-12 lg:gap-24">
					<div className="relative space-y-4">
						<p className="text-muted-foreground">
							Adaptive is evolving to be more than just the models.{" "}
							<span className="font-bold text-accent-foreground">
								It supports an entire ecosystem
							</span>{" "}
							— from products to platforms that help teams scale.
						</p>

						<p className="text-muted-foreground">
							It supports an entire ecosystem — from intuitive interfaces to
							powerful APIs helping developers and businesses adapt to changing
							needs
						</p>

						<div className="grid grid-cols-2 gap-3 pt-6 sm:gap-4">
							<div className="space-y-3">
								<div className="flex items-center gap-2">
									<Zap className="size-4" />
									<h3 className="font-medium text-sm">Lightning Fast</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Experience unparalleled speed with our optimized platform.
								</p>
							</div>
							<div className="space-y-2">
								<div className="flex items-center gap-2">
									<Cpu className="size-4" />
									<h3 className="font-medium text-sm">Powerful</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Leverage advanced capabilities to transform your workflow.
								</p>
							</div>
						</div>
					</div>

					<div className="relative mt-6 sm:mt-0">
						<div className="rounded-2xl bg-[linear-gradient(to_bottom,var(--color-input),transparent)] p-px dark:bg-[linear-gradient(to_bottom,var(--color-card),transparent)]">
							<LogoCarousel logos={logos1} columnCount={2} />
						</div>
						<div className="">
							<LogoCarousel logos={logos2} columnCount={2} />
						</div>
					</div>
				</div>
			</div>
		</section>
	);
}
