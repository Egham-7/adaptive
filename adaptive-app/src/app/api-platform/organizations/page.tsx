"use client";

import { Building2, ChevronRight, Search, Users } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

// Mock data for organizations
const organizations = [
	{
		id: "1",
		name: "Acme AI Solutions",
		description:
			"Leading AI development company focused on enterprise solutions and machine learning platforms.",
		memberCount: 24,
		projectCount: 8,
		role: "Admin",
		color: "blue",
	},
	{
		id: "2",
		name: "TechCorp Research",
		description:
			"Research and development organization specializing in natural language processing and computer vision.",
		memberCount: 12,
		projectCount: 5,
		role: "Developer",
		color: "emerald",
	},
	{
		id: "3",
		name: "StartupLab",
		description:
			"Innovation hub for AI startups and experimental projects in machine learning and automation.",
		memberCount: 8,
		projectCount: 12,
		role: "Contributor",
		color: "orange",
	},
	{
		id: "4",
		name: "DataScience Collective",
		description:
			"Community-driven organization focused on open-source AI tools and collaborative research projects.",
		memberCount: 156,
		projectCount: 23,
		role: "Member",
		color: "violet",
	},
];

export default function OrganizationsPage() {
	const [searchQuery, setSearchQuery] = useState("");

	const filteredOrganizations = organizations.filter(
		(org) =>
			org.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
			org.description.toLowerCase().includes(searchQuery.toLowerCase()),
	);

	const getRoleBadgeVariant = (role: string) => {
		switch (role) {
			case "Admin":
				return "default";
			case "Developer":
				return "secondary";
			case "Contributor":
				return "outline";
			default:
				return "outline";
		}
	};

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
											<div>
												<CardTitle className="text-lg">{org.name}</CardTitle>
												<Badge
													variant={getRoleBadgeVariant(org.role)}
													className="mt-1"
												>
													{org.role}
												</Badge>
											</div>
										</div>
										<ChevronRight className="h-5 w-5 text-muted-foreground" />
									</div>
								</CardHeader>
								<CardContent className="pt-0">
									<CardDescription className="mb-4 text-muted-foreground">
										{org.description}
									</CardDescription>
									<div className="flex items-center gap-4 text-muted-foreground text-sm">
										<div className="flex items-center gap-1">
											<Users className="h-4 w-4" />
											<span>{org.memberCount} members</span>
										</div>
										<div>
											<span>{org.projectCount} projects</span>
										</div>
									</div>
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
