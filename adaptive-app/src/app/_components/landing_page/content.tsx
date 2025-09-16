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
				<h2 className="relative z-10 max-w-4xl font-medium text-4xl lg:text-5xl">
					For Developers & CTOs: Stop overpaying while waiting weeks for new
					model support.
				</h2>

				<div className="grid gap-6 md:grid-cols-2 md:gap-12 lg:gap-24">
					<div className="relative space-y-4">
						<p className="text-muted-foreground">
							Every other router requires weeks of training data and model
							profiling before they work. We work with GPT-6 the day it launches
							- and save you 60-90% on costs across any provider.
						</p>

						<p className="text-muted-foreground">
							New model support without retraining. Provider-agnostic cost
							optimization. Turn your biggest expense into your competitive
							advantage.
						</p>

						<div className="mt-6 rounded-lg border bg-muted/30 p-4">
							<h3 className="mb-2 font-medium text-foreground text-sm">
								How It Works:
							</h3>
							<p className="text-muted-foreground text-sm">
								Our{" "}
								<span className="font-medium text-foreground">
									sub-2ms prompt complexity classifier
								</span>{" "}
								analyzes task complexity, tool calling needs, and model
								capabilities in real-time. When new models launch, we instantly
								map their definitions (tasks, complexity, tools) to our routing
								system - no weeks of training data required.
							</p>
						</div>

						<div className="grid gap-4 pt-6">
							{/* Developer Benefits */}
							<div className="rounded-lg border bg-primary/5 p-4 dark:bg-primary/10">
								<div className="mb-2 flex items-center gap-2">
									<Zap className="size-4 text-primary" />
									<h3 className="font-medium text-foreground text-sm">
										For Developers
									</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Skip weeks of integration work. Use new models day-one with
									existing code. OpenAI-compatible API means zero code changes.
								</p>
							</div>

							{/* CTO Benefits */}
							<div className="rounded-lg border bg-secondary/30 p-4 dark:bg-secondary/20">
								<div className="mb-2 flex items-center gap-2">
									<Cpu className="size-4 text-secondary-foreground" />
									<h3 className="font-medium text-foreground text-sm">
										For CTOs
									</h3>
								</div>
								<p className="text-muted-foreground text-sm">
									Reduce AI spend by 60-90% without vendor lock-in. Get
									competitive advantage while other companies wait for router
									updates.
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
