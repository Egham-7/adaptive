"use client";

import { useUser } from "@clerk/nextjs";
import { AlertCircle, CheckCircle, DollarSign, Users } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/trpc/react";


export default function PromotionalCreditsAdmin() {
	const { user, isLoaded } = useUser();
	const [refreshKey, setRefreshKey] = useState(0);

	// Get promotional stats
	const {
		data: stats,
		isLoading,
		refetch,
	} = api.admin.getPromotionalStats.useQuery(undefined, {
		enabled: isLoaded && !!user,
		refetchOnWindowFocus: false,
	});

	const refresh = () => {
		setRefreshKey((prev) => prev + 1);
		refetch();
	};

	// Check if user is admin
	if (!isLoaded) {
		return (
			<div className="flex min-h-screen items-center justify-center">
				<div className="h-8 w-8 animate-spin rounded-full border-gray-900 border-b-2" />
			</div>
		);
	}

	// Handle access denied (will be caught by the tRPC query error)
	if (stats === undefined && !isLoading) {
		return (
			<div className="flex min-h-screen items-center justify-center">
				<Card className="w-full max-w-md">
					<CardHeader>
						<CardTitle className="flex items-center gap-2 text-red-600">
							<AlertCircle className="h-5 w-5" />
							Access Denied
						</CardTitle>
					</CardHeader>
					<CardContent>
						<p className="text-muted-foreground">
							You don't have permission to access this admin page.
						</p>
					</CardContent>
				</Card>
			</div>
		);
	}

	if (isLoading) {
		return (
			<div className="flex min-h-screen items-center justify-center">
				<div className="h-8 w-8 animate-spin rounded-full border-gray-900 border-b-2" />
			</div>
		);
	}

	return (
		<div className="container mx-auto space-y-6 p-6">
			<div className="flex items-center justify-between">
				<div>
					<h1 className="font-bold text-3xl">Promotional Credits Admin</h1>
					<p className="text-muted-foreground">
						Monitor and manage promotional credit distribution
					</p>
				</div>
				<Button onClick={refresh} variant="outline">
					Refresh Data
				</Button>
			</div>

			{/* Stats Overview */}
			<div className="grid grid-cols-1 gap-4 md:grid-cols-4">
				<Card>
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="font-medium text-sm">Credits Used</CardTitle>
						<Users className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="font-bold text-2xl">{stats?.stats.used || 0}</div>
						<p className="text-muted-foreground text-xs">
							out of {stats?.config.MAX_PROMOTIONAL_USERS} total
						</p>
					</CardContent>
				</Card>

				<Card>
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="font-medium text-sm">Remaining</CardTitle>
						<CheckCircle className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="font-bold text-2xl">
							{stats?.stats.remaining || 0}
						</div>
						<p className="text-muted-foreground text-xs">credits available</p>
					</CardContent>
				</Card>

				<Card>
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="font-medium text-sm">Credit Amount</CardTitle>
						<DollarSign className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="font-bold text-2xl">
							${stats?.config.FREE_CREDIT_AMOUNT.toFixed(2) || "0.00"}
						</div>
						<p className="text-muted-foreground text-xs">per new user</p>
					</CardContent>
				</Card>

				<Card>
					<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
						<CardTitle className="font-medium text-sm">Status</CardTitle>
						<AlertCircle className="h-4 w-4 text-muted-foreground" />
					</CardHeader>
					<CardContent>
						<div className="flex items-center gap-2">
							<Badge variant={stats?.stats.available ? "default" : "secondary"}>
								{stats?.stats.available ? "Active" : "Exhausted"}
							</Badge>
						</div>
						<p className="text-muted-foreground text-xs">
							{stats?.config.ENABLED ? "System enabled" : "System disabled"}
						</p>
					</CardContent>
				</Card>
			</div>

			{/* Transaction History */}
			<Card>
				<CardHeader>
					<CardTitle>Promotional Credit History</CardTitle>
					<p className="text-muted-foreground text-sm">
						All users who have received promotional credits
					</p>
				</CardHeader>
				<CardContent>
					{stats?.transactions.length === 0 ? (
						<p className="py-8 text-center text-muted-foreground">
							No promotional credits awarded yet
						</p>
					) : (
						<div className="space-y-4">
							{stats?.transactions.map((transaction: any) => (
								<div
									key={transaction.userId}
									className="flex items-center justify-between rounded-lg border p-4"
								>
									<div className="flex-1">
										<div className="flex items-center gap-3">
											<Badge variant="outline">#{transaction.userNumber}</Badge>
											<div>
												<p className="font-medium">
													{transaction.organizationName}
												</p>
												<p className="text-muted-foreground text-sm">
													User ID: {transaction.userId}
												</p>
												<p className="text-muted-foreground text-sm">
													Org ID: {transaction.organizationId}
												</p>
											</div>
										</div>
									</div>
									<div className="text-right">
										<p className="font-medium text-green-600">
											+${transaction.amount.toFixed(2)}
										</p>
										<p className="text-muted-foreground text-sm">
											{new Date(transaction.awardedAt).toLocaleDateString()}
										</p>
									</div>
								</div>
							))}
						</div>
					)}
				</CardContent>
			</Card>

			{/* Configuration */}
			<Card>
				<CardHeader>
					<CardTitle>Current Configuration</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="grid grid-cols-1 gap-4 md:grid-cols-2">
						<div>
							<p className="font-medium text-sm">Free Credit Amount</p>
							<p className="font-bold text-2xl">
								${stats?.config.FREE_CREDIT_AMOUNT.toFixed(2)}
							</p>
						</div>
						<div>
							<p className="font-medium text-sm">Max Users</p>
							<p className="font-bold text-2xl">
								{stats?.config.MAX_PROMOTIONAL_USERS}
							</p>
						</div>
						<div>
							<p className="font-medium text-sm">Description</p>
							<p className="text-muted-foreground text-sm">
								{stats?.config.DESCRIPTION}
							</p>
						</div>
						<div>
							<p className="font-medium text-sm">System Status</p>
							<Badge variant={stats?.config.ENABLED ? "default" : "secondary"}>
								{stats?.config.ENABLED ? "Enabled" : "Disabled"}
							</Badge>
						</div>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
