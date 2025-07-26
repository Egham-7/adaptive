"use client";
import { SignUpButton, useUser } from "@clerk/nextjs";
import { Check, Zap, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import SubscribeButton from "../stripe/subscribe-button";
import { api } from "@/trpc/react";
import Link from "next/link";

export default function ChatbotPricing() {
	const { user } = useUser();
	const { data: subscriptionData, isLoading } = api.subscription.getSubscription.useQuery(
		undefined,
		{ enabled: !!user }
	);


	return (
		<div className="w-full p-6">
			<h2 className="mb-8 text-center font-bold text-2xl">Choose Your Plan</h2>
			<div className="mx-auto grid w-full gap-6 md:grid-cols-2">
				<Card>
					<CardHeader>
						<CardTitle className="font-medium">Free</CardTitle>
						<div className="my-3 flex items-baseline">
							<span className="font-semibold text-2xl">$0</span>
							<span className="ml-1 text-muted-foreground text-sm">/month</span>
						</div>
						<CardDescription className="text-sm">
							Perfect for trying out our chatbot
						</CardDescription>
						{!user ? (
							<Button asChild variant="outline" className="mt-4 w-full">
								<SignUpButton />
							</Button>
						) : (
							<Button disabled variant="outline" className="mt-4 w-full">
								Signed Up
							</Button>
						)}
					</CardHeader>
					<CardContent className="space-y-4">
						<hr className="border-dashed" />
						<ul className="list-outside space-y-3 text-sm">
							{[
								"7 messages per day",
								"Basic AI capabilities",
								"Community support",
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
							<span className="ml-1 text-muted-foreground text-sm">/month</span>
						</div>
						<CardDescription className="text-sm">
							For unlimited chatbot usage
						</CardDescription>
						{isLoading ? (
							<Button disabled className="mt-4 w-full bg-muted text-muted-foreground">
								Loading...
							</Button>
						) : !user ? (
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
						) : !subscriptionData?.subscribed ? (
							<div className="mt-4 w-full">
								<SubscribeButton />
							</div>
						) : (
							<Button asChild className="mt-4 w-full">
								<Link href="/billing">
									<Settings className="relative mr-2 size-4" />
									<span>Manage Subscription</span>
								</Link>
							</Button>
						)}
					</CardHeader>
					<CardContent className="space-y-4">
						<hr className="border-dashed" />
						<ul className="list-outside space-y-3 text-sm">
							{[
								"Unlimited messages",
								"Advanced AI capabilities",
								"Priority support",
								"API access",
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
		</div>
	);
}
