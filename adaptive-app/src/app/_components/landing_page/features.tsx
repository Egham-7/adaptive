import { Shield, TrendingDown, Zap } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { ProviderDownloadChart } from "./provider-download-chart";

export default function FeaturesSection() {
	return (
		<section
			id="features"
			className="bg-muted/50 py-16 md:py-32 dark:bg-transparent"
		>
			<div className="mx-auto max-w-3xl px-6 lg:max-w-5xl">
				<header className="mb-12 text-center">
					<h2 className="font-display font-semibold text-4xl lg:text-5xl">
						New Models Day One. Any Provider. Maximum Savings.
					</h2>
					<p className="mt-4 text-muted-foreground">
						The only router that works with tomorrow's models today - no
						training data, no vendor lock-in, 60-90% savings from day one
					</p>
				</header>
				<div className="grid gap-8 md:grid-cols-3">
					{/* Pillar 1: New Models Day One */}
					<Card className="relative overflow-hidden">
						<CardContent className="pt-8 text-center">
							<div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-primary/10">
								<Zap className="h-10 w-10 text-primary" aria-hidden="true" />
							</div>
							<h3 className="mb-4 font-semibold text-2xl">
								New Models Day One
							</h3>
							<p className="text-muted-foreground text-sm leading-relaxed">
								Works with GPT-6, Claude Opus 5, or any new model the day it
								launches. Our{" "}
								<span className="font-medium text-foreground">
									sub-2ms classifier
								</span>{" "}
								instantly maps model definitions - no training data, no weeks of
								setup.
							</p>
							<div className="mt-6 rounded-lg bg-muted/50 p-3">
								<p className="font-mono text-muted-foreground text-xs">
									&lt; 2ms inference time
								</p>
							</div>
						</CardContent>
					</Card>

					{/* Pillar 2: Any Provider */}
					<Card className="relative overflow-hidden">
						<CardContent className="pt-8 text-center">
							<div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-primary/10">
								<TrendingDown className="h-10 w-10 text-primary" />
							</div>
							<h3 className="mb-4 font-semibold text-2xl">Any Provider</h3>
							<p className="text-muted-foreground text-sm leading-relaxed">
								Provider-agnostic optimization across OpenAI, Anthropic, Google,
								and any future provider. Never locked into expensive routing
								decisions. Switch providers seamlessly.
							</p>
							<div className="mt-4">
								<ProviderDownloadChart />
							</div>
						</CardContent>
					</Card>

					{/* Pillar 3: Maximum Savings */}
					<Card className="relative overflow-hidden">
						<CardContent className="pt-8 text-center">
							<div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-primary/10">
								<Shield className="h-10 w-10 text-primary" />
							</div>
							<h3 className="mb-4 font-semibold text-2xl">Maximum Savings</h3>
							<p className="text-muted-foreground text-sm leading-relaxed">
								Achieve 60-90% cost reduction through intelligent routing and
								real-time optimization. Sub-millisecond decisions with automatic
								failover ensure 99.9% uptime.
							</p>
							<div className="mt-6 space-y-2">
								<div className="flex justify-between text-xs">
									<span className="text-muted-foreground">Average Savings</span>
									<span className="font-medium text-primary">60-90%</span>
								</div>
								<div className="flex justify-between text-xs">
									<span className="text-muted-foreground">Routing Speed</span>
									<span className="font-medium text-primary">&lt; 2ms</span>
								</div>
							</div>
						</CardContent>
					</Card>
				</div>
			</div>
		</section>
	);
}
