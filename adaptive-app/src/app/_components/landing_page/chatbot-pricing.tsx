"use client";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { SignUpButton, useUser } from "@clerk/nextjs";
import { Check, Zap } from "lucide-react";
import SubscribeButton from "../stripe/subscribe-button";

export default function ChatbotPricing() {
	const { user } = useUser();

	return (
		<section className="overflow-hidden py-16 md:py-32">
			<div className="mx-auto max-w-6xl px-6">
				<div className="mx-auto max-w-2xl space-y-6 text-center">
					<h1 className="text-balance text-center font-display font-semibold text-4xl lg:text-5xl">
						Chatbot Pricing
					</h1>
					<p className="text-muted-foreground">
						Simple, Transparent Pricing. Choose the perfect plan for your
						chatbot needs. All plans include our core features with no hidden
						fees.
					</p>
				</div>

				<div className="mt-8 grid gap-6 md:mt-20 md:grid-cols-2">
					<Card>
						<CardHeader>
							<CardTitle className="font-medium">Free</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">$0</span>
								<span className="ml-1 text-muted-foreground text-sm">
									/month
								</span>
							</div>
							<CardDescription className="text-sm">
								Perfect for trying out our chatbot
							</CardDescription>
							<Button asChild variant="outline" className="mt-4 w-full">
								<SignUpButton />
							</Button>
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"50 messages per month",
									"Basic AI capabilities",
									"Standard response time",
									"Community support",
									"Basic analytics",
									"1 chatbot instance",
								].map((item) => (
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
							<CardTitle className="font-medium">Pro</CardTitle>
							<div className="my-3 flex items-baseline">
								<span className="font-semibold text-2xl">$4.99</span>
								<span className="ml-1 text-muted-foreground text-sm">
									/month
								</span>
							</div>
							<CardDescription className="text-sm">
								For unlimited chatbot usage
							</CardDescription>
							{user ? (
								<div className="mt-4 w-full">
									<SubscribeButton userId={user.id} />
								</div>
							) : (
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
							)}
						</CardHeader>
						<CardContent className="space-y-4">
							<hr className="border-dashed" />
							<ul className="list-outside space-y-3 text-sm">
								{[
									"Unlimited messages",
									"Advanced AI capabilities",
									"Priority response time",
									"Email support",
									"Advanced analytics",
									"1 chatbot instance",
									"Custom branding",
									"API access",
									"99.9% uptime SLA",
								].map((item) => (
									<li key={item} className="flex items-center gap-2">
										<Check className="size-3 text-primary" />
										{item}
									</li>
								))}
							</ul>
						</CardContent>
					</Card>
				</div>

				<div className="mx-auto mt-16 max-w-3xl">
					<div className="rounded-lg border bg-muted/30 p-6">
						<h3 className="mb-4 font-medium text-lg">All Plans Include</h3>
						<div className="grid grid-cols-1 gap-4 md:grid-cols-3">
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									Core Features
								</div>
								<div className="mt-2 text-sm">Basic chatbot functionality</div>
							</div>
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									Security
								</div>
								<div className="mt-2 text-sm">Data encryption & privacy</div>
							</div>
							<div className="rounded-md border bg-background p-4">
								<div className="font-medium text-muted-foreground text-sm">
									Updates
								</div>
								<div className="mt-2 text-sm">Regular feature updates</div>
							</div>
						</div>
						<p className="mt-4 text-muted-foreground text-sm">
							Start with our free plan and upgrade anytime to unlock unlimited
							messages and premium features.
						</p>
					</div>
				</div>
			</div>
		</section>
	);
}
