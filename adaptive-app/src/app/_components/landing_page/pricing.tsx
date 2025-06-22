import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { SignUpButton } from "@clerk/nextjs";
import { Check, Zap } from "lucide-react";
import Link from "next/link";

export default function Pricing() {
	return (
		<section className="overflow-hidden py-16 md:py-32">
			<div className="mx-auto max-w-6xl px-6">
				<div className="mx-auto max-w-2xl space-y-6 text-center">
					<h1 className="text-balance text-center font-display font-semibold text-4xl lg:text-5xl">
						Pay Only for What You Use
					</h1>
					<p className="text-muted-foreground">
						Adaptive&apos;s usage-based pricing ensures you only pay for the
						resources you consume. Scale up or down instantly with no upfront
						commitments or hidden fees.
					</p>
				</div>

				<div className="mt-8 grid gap-6 md:mt-20 md:grid-cols-3">
					<Card>
						<CardHeader>
							<CardTitle className="font-medium">Developer</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">$0.005</span>
								<span className="ml-1 text-muted-foreground text-sm">
									/ 1K tokens
								</span>
							</div>
							<CardDescription className="text-sm">
								No minimum spend, pay as you go
							</CardDescription>
							<Button asChild variant="outline" className="mt-4 w-full">
								<SignUpButton />
							</Button>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Pay-as-you-go pricing",
									"Basic LLM routing",
									"Standard API access",
									"Community support",
									"$10 free credit to start",
									"No credit card required",
								].map((item, index) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>

					<Card className="relative">
						<span className="-top-3 absolute inset-x-0 mx-auto flex h-6 w-fit items-center rounded-full bg-[linear-gradient(to_right,var(--color-primary),var(--color-secondary))] px-3 py-1 font-medium text-primary-foreground text-xs ring-1 ring-white/20 ring-inset ring-offset-1 ring-offset-gray-950/5">
							Popular
						</span>
						<CardHeader>
							<CardTitle className="font-medium">Business</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">$0.004</span>
								<span className="ml-1 text-muted-foreground text-sm">
									/ 1K tokens
								</span>
							</div>
							<CardDescription className="text-sm">
								$100 minimum monthly spend
							</CardDescription>
							<Button
								asChild
								className="mt-4 w-full bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
							>
								<SignUpButton>
									<Button>
										<Zap className="relative mr-2 size-4" />
										<span>Get Started</span>
									</Button>
								</SignUpButton>
							</Button>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"20% volume discount",
									"Advanced LLM routing",
									"Multi-model fallbacks",
									"Priority email support",
									"Usage analytics dashboard",
									"Custom API keys",
									"Request caching",
									"Webhook integrations",
									"Team access controls",
									"99.9% uptime SLA",
								].map((item, index) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>

					<Card className="flex flex-col">
						
						<CardHeader>
							<CardTitle className="font-medium">Enterprise</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">Custom</span>
							</div>
							<CardDescription className="text-sm">
								Volume-based discounts available
							</CardDescription>
							<Button
								asChild
								variant="outline"
								className="mt-4 w-full border-primary text-primary hover:bg-primary/10 dark:hover:bg-primary/20"
							>
								<Link href="/">Contact Sales</Link>
							</Button>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Volume-based pricing",
									"Dedicated account manager",
									"Custom SLAs and support",
									"Advanced cost optimization",
									"Private model deployment",
									"Custom model fine-tuning",
									"Enterprise SSO",
									"Audit logs & compliance",
									"On-premise deployment options",
									"MSA and custom contracts",
								].map((item, index) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>
				</div>

				<div className="mx-auto mt-16 max-w-3xl ">
					<div className="rounded-xl border bg-card p-6">
						<h3 className="mb-4 font-medium text-lg">Volume Discounts</h3>
						<div className="grid grid-cols-1 gap-4 md:grid-cols-3">
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									1M - 10M tokens/month
								</div>
								<div className="mt-2 font-semibold text-xl">10% off</div>
							</div>
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									10M - 50M tokens/month
								</div>
								<div className="mt-2 font-semibold text-xl">20% off</div>
							</div>
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									50M+ tokens/month
								</div>
								<div className="mt-2 font-semibold text-xl">Contact us</div>
							</div>
						</div>
						<p className="mt-4 text-muted-foreground text-sm">
							All plans include access to our core infrastructure. Volume
							discounts are applied automatically based on your monthly usage.
						</p>
					</div>
				</div>
			</div>
		</section>
	);
}
