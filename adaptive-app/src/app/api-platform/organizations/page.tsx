"use client";

import { Building2, ChevronRight, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useOrganizations } from "@/hooks/organizations/use-organizations";

export default function OrganizationsPage() {
	const [searchQuery, setSearchQuery] = useState("");
	const { data: organizations = [] } = useOrganizations();

	const filteredOrganizations = organizations.filter(
		(org) =>
			org.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
			org.description?.toLowerCase().includes(searchQuery.toLowerCase()),
	);

	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto max-w-6xl px-4 py-8">
				<div className="mb-8">
					<h1 className="mb-2 font-bold text-3xl text-foreground">
						Organizations
					</h1>
					<p className="text-muted-foreground">
						Manage your AI projects and collaborate with teams
					</p>
				</div>

				<div className="mb-8">
					<div className="relative max-w-md">
						<Search className="-translate-y-1/2 absolute top-1/2 left-3 h-4 w-4 transform text-muted-foreground" />
						<Input
							placeholder="Search organizations..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="pl-10"
						/>
					</div>
				</div>

				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
					{filteredOrganizations.map((org) => (
						<Card key={org.id} className="transition-shadow hover:shadow-md">
							<Link href={`/api-platform/organizations/${org.id}`}>
								<CardHeader className="pb-4">
									<div className="flex items-start justify-between">
										<div className="flex items-center gap-3">
											<div className="rounded-lg bg-muted p-2">
												<Building2 className="h-5 w-5 text-muted-foreground" />
											</div>
										</div>
										<ChevronRight className="h-5 w-5 text-muted-foreground" />
									</div>
								</CardHeader>
								<CardContent className="pt-0">
									<CardDescription className="mb-4 text-muted-foreground">
										{org.description}
									</CardDescription>
								</CardContent>
							</Link>
						</Card>
					))}
				</div>

				{filteredOrganizations.length === 0 && (
					<div className="py-12 text-center">
						<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-muted p-4">
							<Building2 className="h-8 w-8 text-muted-foreground" />
						</div>
						<h3 className="mb-2 font-semibold text-foreground text-lg">
							No organizations found
						</h3>
						<p className="mx-auto max-w-md text-muted-foreground">
							{searchQuery
								? "Try adjusting your search terms"
								: "You don't belong to any organizations yet"}
						</p>
					</div>
				)}
			</div>
		</div>
	);
}
