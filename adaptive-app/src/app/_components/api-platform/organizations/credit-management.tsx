"use client";

import { useState } from "react";
import { CreditCard, DollarSign, TrendingUp, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { api } from "@/trpc/react";
import { toast } from "sonner";

interface CreditManagementProps {
	organizationId: string;
}

const CREDIT_PACKAGES = [
	{
		id: "starter",
		name: "Starter Pack",
		amount: 10,
		price: 10,
		description: "Perfect for small projects",
		popular: false,
	},
	{
		id: "professional",
		name: "Professional",
		amount: 50,
		price: 50,
		description: "Most popular for growing teams",
		popular: true,
	},
	{
		id: "enterprise",
		name: "Enterprise",
		amount: 200,
		price: 200,
		description: "For large-scale operations",
		popular: false,
	},
];

export function CreditManagement({ organizationId }: CreditManagementProps) {
	const [customAmount, setCustomAmount] = useState<string>("");
	const [showCustomDialog, setShowCustomDialog] = useState(false);
	const [isProcessing, setIsProcessing] = useState(false);

	// Fetch credit balance
	const { data: balance, refetch: refetchBalance } = api.credits.getBalance.useQuery(
		{ organizationId },
		{ refetchInterval: 30000 } // Refetch every 30 seconds
	);

	// Fetch low balance status
	const { data: balanceStatus } = api.credits.getLowBalanceStatus.useQuery(
		{ organizationId }
	);

	// Create checkout session
	const createCheckout = api.credits.createCheckoutSession.useMutation({
		onSuccess: (data) => {
			if (data.checkoutUrl) {
				window.location.href = data.checkoutUrl;
			}
		},
		onError: (error) => {
			toast.error("Failed to create checkout session", {
				description: error.message,
			});
			setIsProcessing(false);
		},
	});

	const handlePurchase = async (amount: number) => {
		setIsProcessing(true);
		
		try {
			await createCheckout.mutateAsync({
				organizationId,
				amount,
				successUrl: `${window.location.origin}/api-platform/organizations/${organizationId}?purchase=success`,
				cancelUrl: `${window.location.origin}/api-platform/organizations/${organizationId}?purchase=cancelled`,
			});
		} catch (error) {
			setIsProcessing(false);
		}
	};

	const handleCustomPurchase = () => {
		const amount = parseFloat(customAmount);
		if (amount >= 1 && amount <= 10000) {
			handlePurchase(amount);
			setShowCustomDialog(false);
		} else {
			toast.error("Please enter an amount between $1 and $10,000");
		}
	};

	const getBalanceStatusColor = (status?: string) => {
		switch (status) {
			case "empty":
				return "text-destructive";
			case "very_low":
				return "text-orange-600";
			case "low":
				return "text-yellow-600";
			default:
				return "text-green-600";
		}
	};

	const getBalanceStatusIcon = (status?: string) => {
		switch (status) {
			case "empty":
			case "very_low":
				return "⚠️";
			case "low":
				return "⚡";
			default:
				return "✅";
		}
	};

	return (
		<div className="space-y-6">
			{/* Credit Balance Overview */}
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2">
						<CreditCard className="h-5 w-5" />
						Credit Balance
					</CardTitle>
					<CardDescription>
						Monitor your API usage credits and purchase more when needed
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="flex items-center justify-between">
						<div className="space-y-1">
							<div className="flex items-center gap-2">
								<span className="text-2xl font-bold">
									{balance?.formattedBalance || "$0.00"}
								</span>
								{balanceStatus?.status && (
									<span className={getBalanceStatusColor(balanceStatus.status)}>
										{getBalanceStatusIcon(balanceStatus.status)}
									</span>
								)}
							</div>
							{balanceStatus?.message && (
								<p className={`text-sm ${getBalanceStatusColor(balanceStatus.status)}`}>
									{balanceStatus.message}
								</p>
							)}
						</div>
						<Button 
							onClick={() => refetchBalance()}
							variant="outline"
							size="sm"
						>
							Refresh
						</Button>
					</div>

					{balanceStatus?.status && ["empty", "very_low", "low"].includes(balanceStatus.status) && (
						<div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
							<p className="text-orange-800 text-sm font-medium">
								💡 Low credit balance detected. Consider purchasing more credits to avoid service interruption.
							</p>
						</div>
					)}
				</CardContent>
			</Card>

			{/* Quick Purchase Options */}
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center gap-2">
						<DollarSign className="h-5 w-5" />
						Purchase Credits
					</CardTitle>
					<CardDescription>
						Choose from our credit packages or enter a custom amount
					</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="grid gap-4 md:grid-cols-3">
						{CREDIT_PACKAGES.map((pkg) => (
							<Card key={pkg.id} className="relative border-2 hover:border-primary/20 transition-colors">
								{pkg.popular && (
									<Badge className="absolute -top-2 left-1/2 -translate-x-1/2">
										Most Popular
									</Badge>
								)}
								<CardHeader className="text-center pb-2">
									<CardTitle className="text-lg">{pkg.name}</CardTitle>
									<CardDescription>{pkg.description}</CardDescription>
								</CardHeader>
								<CardContent className="text-center space-y-4">
									<div>
										<div className="text-3xl font-bold">${pkg.amount}</div>
										<div className="text-sm text-muted-foreground">
											${pkg.price} total
										</div>
									</div>
									<Button 
										onClick={() => handlePurchase(pkg.amount)}
										disabled={isProcessing}
										className="w-full"
										variant={pkg.popular ? "default" : "outline"}
									>
										{isProcessing ? "Processing..." : "Purchase"}
									</Button>
								</CardContent>
							</Card>
						))}
					</div>

					<Separator className="my-6" />

					{/* Custom Amount */}
					<div className="text-center">
						<Dialog open={showCustomDialog} onOpenChange={setShowCustomDialog}>
							<DialogTrigger asChild>
								<Button variant="outline" className="w-full max-w-xs">
									<Zap className="mr-2 h-4 w-4" />
									Custom Amount
								</Button>
							</DialogTrigger>
							<DialogContent className="max-w-md">
								<DialogHeader>
									<DialogTitle>Purchase Custom Amount</DialogTitle>
								</DialogHeader>
								<div className="space-y-4">
									<div>
										<Label htmlFor="amount">Amount (USD)</Label>
										<Input
											id="amount"
											type="number"
											min="1"
											max="10000"
											step="0.01"
											placeholder="Enter amount..."
											value={customAmount}
											onChange={(e) => setCustomAmount(e.target.value)}
										/>
										<p className="text-sm text-muted-foreground mt-1">
											Minimum: $1.00 • Maximum: $10,000.00
										</p>
									</div>
									<div className="flex gap-2">
										<Button
											variant="outline"
											onClick={() => setShowCustomDialog(false)}
											className="flex-1"
										>
											Cancel
										</Button>
										<Button
											onClick={handleCustomPurchase}
											disabled={isProcessing || !customAmount}
											className="flex-1"
										>
											{isProcessing ? "Processing..." : "Purchase"}
										</Button>
									</div>
								</div>
							</DialogContent>
						</Dialog>
					</div>

					{/* Pricing Info */}
					<div className="mt-6 p-4 bg-muted/50 rounded-lg">
						<h4 className="font-medium text-sm mb-2 flex items-center gap-2">
							<TrendingUp className="h-4 w-4" />
							API Pricing
						</h4>
						<div className="grid gap-2 text-sm">
							<div className="flex justify-between">
								<span className="text-muted-foreground">Input tokens (per 1M)</span>
								<span className="font-mono">$0.05</span>
							</div>
							<div className="flex justify-between">
								<span className="text-muted-foreground">Output tokens (per 1M)</span>
								<span className="font-mono">$0.15</span>
							</div>
						</div>
						<p className="text-xs text-muted-foreground mt-2">
							Credits are charged based on actual token usage. 1 credit = $1 USD
						</p>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}