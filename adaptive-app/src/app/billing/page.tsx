"use client";

import { useState } from "react";
import { useUser } from "@clerk/nextjs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { CalendarDays, CreditCard, AlertTriangle } from "lucide-react";
import { api } from "@/trpc/react";
import { useRouter } from "next/navigation";
import { useSubscription } from "@/hooks/use-subscription";

export default function BillingPage() {
	const { user } = useUser();
	const router = useRouter();
	const [showCancelDialog, setShowCancelDialog] = useState(false);

	const { data: subscriptionData, isLoading } = api.subscription.getSubscription.useQuery(
		undefined,
		{ enabled: !!user }
	);

	const { cancelSubscription } = useSubscription();

	if (!user) {
		router.push("/");
		return null;
	}

	if (isLoading) {
		return (
			<div className="container max-w-4xl mx-auto p-6">
				<div className="space-y-6">
					<div className="h-8 bg-muted rounded animate-pulse" />
					<div className="h-32 bg-muted rounded animate-pulse" />
				</div>
			</div>
		);
	}

	const subscription = subscriptionData?.subscription;
	const isSubscribed = subscriptionData?.subscribed;

	return (
		<div className="container max-w-4xl mx-auto p-6">
			<div className="space-y-6">
				<div>
					<h1 className="text-3xl font-bold">Billing & Subscription</h1>
					<p className="text-muted-foreground">Manage your subscription and billing preferences</p>
				</div>

				{/* Current Plan Card */}
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center gap-2">
							<CreditCard className="h-5 w-5" />
							Current Plan
						</CardTitle>
					</CardHeader>
					<CardContent className="space-y-4">
						{isSubscribed ? (
							<>
								<div className="flex items-center justify-between">
									<div>
										<h3 className="font-semibold text-lg">Pro Plan</h3>
										<p className="text-muted-foreground">Unlimited messages and advanced features</p>
									</div>
									<Badge variant="default">Active</Badge>
								</div>
								
								<div className="flex items-center gap-2 text-sm text-muted-foreground">
									<CalendarDays className="h-4 w-4" />
									<span>
										Next billing: {subscription?.currentPeriodEnd ? 
											new Date(subscription.currentPeriodEnd).toLocaleDateString() : 
											"N/A"
										}
									</span>
								</div>

								<div className="pt-4 border-t">
									<h4 className="font-medium mb-2">Plan Features</h4>
									<ul className="space-y-1 text-sm text-muted-foreground">
										<li>• Unlimited messages</li>
										<li>• Advanced AI capabilities</li>
										<li>• Priority support</li>
										<li>• API access</li>
									</ul>
								</div>
							</>
						) : (
							<>
								<div className="flex items-center justify-between">
									<div>
										<h3 className="font-semibold text-lg">Free Plan</h3>
										<p className="text-muted-foreground">Basic features with daily limits</p>
									</div>
									<Badge variant="secondary">Free</Badge>
								</div>
								
								<div className="pt-4 border-t">
									<h4 className="font-medium mb-2">Plan Features</h4>
									<ul className="space-y-1 text-sm text-muted-foreground">
										<li>• 7 messages per day</li>
										<li>• Basic AI capabilities</li>
										<li>• Community support</li>
									</ul>
								</div>

								<Button 
									onClick={() => router.push("/#pricing")}
									className="w-full mt-4"
								>
									Upgrade to Pro
								</Button>
							</>
						)}
					</CardContent>
				</Card>

				{/* Billing Actions */}
				{isSubscribed && (
					<Card>
						<CardHeader>
							<CardTitle>Subscription Management</CardTitle>
							<CardDescription>
								Manage your subscription settings and billing preferences
							</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							<div className="flex flex-col sm:flex-row gap-4">
								<Button variant="outline" className="flex-1">
									Update Payment Method
								</Button>
								<Button variant="outline" className="flex-1">
									Download Invoices
								</Button>
							</div>
							
							<div className="pt-4 border-t">
								<Dialog open={showCancelDialog} onOpenChange={setShowCancelDialog}>
									<DialogTrigger asChild>
										<Button variant="destructive" className="w-full">
											Cancel Subscription
										</Button>
									</DialogTrigger>
									<DialogContent>
										<DialogHeader>
											<DialogTitle className="flex items-center gap-2">
												<AlertTriangle className="h-5 w-5 text-destructive" />
												Cancel Subscription
											</DialogTitle>
											<DialogDescription>
												Are you sure you want to cancel your subscription? This action will:
											</DialogDescription>
										</DialogHeader>
										
										<div className="space-y-4">
											<div className="bg-muted/50 p-4 rounded-lg">
												<ul className="space-y-2 text-sm">
													<li>• Your subscription will remain active until {subscription?.currentPeriodEnd ? 
														new Date(subscription.currentPeriodEnd).toLocaleDateString() : 
														"the end of your billing period"
													}</li>
													<li>• You'll lose access to Pro features after this date</li>
													<li>• Your account will revert to the Free plan</li>
													<li>• You can resubscribe at any time</li>
												</ul>
											</div>
										</div>

										<DialogFooter>
											<Button 
												variant="outline" 
												onClick={() => setShowCancelDialog(false)}
											>
												Keep Subscription
											</Button>
											<Button 
												variant="destructive"
												onClick={() => {
													cancelSubscription.mutate();
													setShowCancelDialog(false);
												}}
												disabled={cancelSubscription.isPending}
											>
												{cancelSubscription.isPending ? "Canceling..." : "Yes, Cancel"}
											</Button>
										</DialogFooter>
									</DialogContent>
								</Dialog>
							</div>
						</CardContent>
					</Card>
				)}
			</div>
		</div>
	);
}