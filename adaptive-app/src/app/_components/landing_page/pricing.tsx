"use client";

import { SignedIn, SignedOut, SignUpButton } from "@clerk/nextjs";
import { Check, Zap } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { useSmartRedirect } from "@/hooks/use-smart-redirect";

export default function Pricing() {
	const [isAnnual, setIsAnnual] = useState(true);
	const redirectPath = useSmartRedirect();

	return (
		<section id="pricing" className="overflow-hidden py-16 md:py-32">
			<div className="mx-auto max-w-6xl px-6">
				<header className="mx-auto max-w-2xl space-y-6 text-center">
					<h2 className="text-balance font-display font-semibold text-4xl lg:text-5xl">
						API Pricing
					</h2>
					<p className="text-muted-foreground">
						Simple, transparent pricing for our AI routing API. Chatbot pricing
						available in the chat app.
					</p>

					<fieldset className="mt-8 flex items-center justify-center gap-4 border-0 p-0">
						<legend className="sr-only">Billing frequency</legend>
						<span
							className={`text-sm ${!isAnnual ? "text-foreground" : "text-muted-foreground"}`}
						>
							Monthly
						</span>
						<button
							type="button"
							onClick={() => setIsAnnual(!isAnnual)}
							aria-label={`Switch to ${isAnnual ? "monthly" : "annual"} billing`}
							aria-pressed={isAnnual}
							className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
								isAnnual ? "bg-primary" : "bg-muted"
							}`}
						>
							<span
								className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
									isAnnual ? "translate-x-6" : "translate-x-1"
								}`}
							/>
						</button>
						<span
							className={`text-sm ${isAnnual ? "text-foreground" : "text-muted-foreground"}`}
						>
							Annual
						</span>
						{isAnnual && (
							<span className="rounded-full border bg-accent px-2 py-1 text-accent-foreground text-xs">
								Save 20%
							</span>
						)}
					</fieldset>
				</header>

				<div className="mx-auto mt-8 grid max-w-6xl gap-6 md:mt-20 md:grid-cols-3">
					<Card>
						<CardHeader>
							<CardTitle className="font-medium">Developer</CardTitle>
							<div className="my-3 space-y-2">
								<div className="flex items-baseline">
									<span className="font-semibold text-2xl">+$0.10</span>
									<span className="ml-1 text-muted-foreground text-sm">
										per 1M tokens
									</span>
								</div>
								<div className="text-muted-foreground text-xs">
									Added to original model pricing (input + output)
								</div>
							</div>
							<CardDescription className="text-sm">
								Simple markup pricing - pay original model cost + $0.10 per
								million tokens. Semantic cache hits: +$0.05/1M tokens. Prompt
								response cache hits: Free!
							</CardDescription>
							<SignedOut>
								<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
									<Button variant="outline" className="mt-4 w-full">
										Get Started
									</Button>
								</SignUpButton>
							</SignedOut>
							<SignedIn>
								{redirectPath ? (
									<Button variant="outline" className="mt-4 w-full" asChild>
										<Link href={redirectPath}>Get Started</Link>
									</Button>
								) : (
									<Button variant="outline" className="mt-4 w-full" disabled>
										Get Started
									</Button>
								)}
							</SignedIn>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Pay-as-you-go pricing",
									"Basic AI routing",
									"Standard API access",
									"Community support",
									"$10 free credit to start",
									"No minimum commitment",
									"Rate limiting: 1K req/hour",
									"Email support",
								].map((item, _index) => (
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
							<CardTitle className="font-medium">Team</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">
									${isAnnual ? "200" : "20"}
								</span>
								<span className="ml-1 text-muted-foreground text-sm">
									/ member / {isAnnual ? "year" : "month"}
								</span>
							</div>
							<CardDescription className="text-sm">
								{isAnnual ? "Annual" : "Monthly"} licensing fee per team member
								{isAnnual && " (20% savings vs monthly)"}
							</CardDescription>
							<SignedOut>
								<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
									<Button className="mt-4 w-full bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90">
										<Zap className="relative mr-2 size-4" />
										<span>Get Started</span>
									</Button>
								</SignUpButton>
							</SignedOut>
							<SignedIn>
								{redirectPath ? (
									<Button
										className="mt-4 w-full bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
										asChild
									>
										<Link href={redirectPath}>
											<Zap className="relative mr-2 size-4" />
											<span>Get Started</span>
										</Link>
									</Button>
								) : (
									<Button
										className="mt-4 w-full bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
										disabled
									>
										<Zap className="relative mr-2 size-4" />
										<span>Get Started</span>
									</Button>
								)}
							</SignedIn>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Everything in Developer",
									"Advanced smart routing",
									"Multi-provider failover",
									"Priority email support",
									"Analytics & observability",
									"Team management dashboard",
									"Custom API keys",
									"Advanced caching",
									"Webhook integrations",
									"99.9% uptime SLA",
									"Higher rate limits (10K req/hour)",
									"Dedicated account manager",
								].map((item, _index) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>

					<Card>
						<CardHeader>
							<CardTitle className="font-medium">Enterprise</CardTitle>
							<div className="my-3 space-y-2">
								<div className="flex items-baseline">
									<span className="font-semibold text-2xl">Custom</span>
								</div>
								<div className="text-muted-foreground text-xs">
									Tailored pricing for your organization
								</div>
							</div>
							<CardDescription className="text-sm">
								Custom pricing based on usage volume and specific requirements
							</CardDescription>
							<Button variant="outline" className="mt-4 w-full">
								Contact Us
							</Button>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Everything in Team",
									"Custom integrations",
									"On-premise deployment",
									"Dedicated infrastructure",
									"24/7 phone support",
									"Custom SLA agreements",
									"Advanced security features",
									"SOC 2 compliance",
									"Custom rate limits",
									"Professional services",
									"Training & onboarding",
									"Custom contract terms",
								].map((item, _index) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>
				</div>
			</div>
		</section>
	);
}
