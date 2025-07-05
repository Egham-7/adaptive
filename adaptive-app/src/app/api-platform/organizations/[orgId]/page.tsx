"use client";

import {
	Activity,
	ArrowLeft,
	Calendar,
	CheckCircle,
	ChevronRight,
	Clock,
	Folder,
	Pause,
	Search,
	Users,
	XCircle,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

// TypeScript interfaces
interface Organization {
	name: string;
	description: string;
	gradient: string;
}

interface Project {
	id: string;
	name: string;
	description: string;
	status: "active" | "inactive" | "paused";
	lastUpdated: string;
	members: number;
	createdAt: string;
	progress: number;
}

// Mock data for organizations and projects
const organizations: Record<string, Organization> = {
	"1": {
		name: "Acme AI Solutions",
		description:
			"Leading AI development company focused on enterprise solutions and machine learning platforms.",
		gradient: "from-blue-500 to-purple-600",
	},
	"2": {
		name: "TechCorp Research",
		description:
			"Research and development organization specializing in natural language processing and computer vision.",
		gradient: "from-emerald-500 to-teal-600",
	},
	"3": {
		name: "StartupLab",
		description:
			"Innovation hub for AI startups and experimental projects in machine learning and automation.",
		gradient: "from-orange-500 to-red-600",
	},
};

const projects: Record<string, Project[]> = {
	"1": [
		{
			id: "1",
			name: "Customer Support AI",
			description:
				"Advanced chatbot system for automated customer service with natural language understanding.",
			status: "active",
			lastUpdated: "2 hours ago",
			members: 6,
			createdAt: "2024-01-15",
			progress: 85,
		},
		{
			id: "2",
			name: "Document Analysis Engine",
			description:
				"ML-powered document processing and information extraction system for enterprise clients.",
			status: "active",
			lastUpdated: "1 day ago",
			members: 4,
			createdAt: "2024-02-01",
			progress: 92,
		},
		{
			id: "3",
			name: "Predictive Analytics Dashboard",
			description:
				"Real-time analytics platform with predictive modeling capabilities for business intelligence.",
			status: "inactive",
			lastUpdated: "1 week ago",
			members: 3,
			createdAt: "2023-12-10",
			progress: 45,
		},
		{
			id: "4",
			name: "Voice Recognition API",
			description:
				"High-accuracy speech-to-text service with multi-language support and real-time processing.",
			status: "paused",
			lastUpdated: "3 days ago",
			members: 8,
			createdAt: "2024-01-20",
			progress: 67,
		},
		{
			id: "5",
			name: "Image Classification Service",
			description:
				"Computer vision API for automated image tagging and content moderation across platforms.",
			status: "active",
			lastUpdated: "4 hours ago",
			members: 5,
			createdAt: "2024-02-15",
			progress: 78,
		},
	],
	"2": [
		{
			id: "6",
			name: "NLP Research Framework",
			description:
				"Open-source framework for natural language processing research and experimentation.",
			status: "active",
			lastUpdated: "6 hours ago",
			members: 12,
			createdAt: "2023-11-01",
			progress: 95,
		},
		{
			id: "7",
			name: "Computer Vision Lab",
			description:
				"Experimental platform for testing new computer vision algorithms and models.",
			status: "active",
			lastUpdated: "1 day ago",
			members: 8,
			createdAt: "2024-01-05",
			progress: 73,
		},
	],
	"3": [
		{
			id: "8",
			name: "AI Startup Accelerator",
			description:
				"Platform connecting AI startups with resources, mentorship, and funding opportunities.",
			status: "active",
			lastUpdated: "30 minutes ago",
			members: 15,
			createdAt: "2024-02-20",
			progress: 88,
		},
	],
};

export default function OrganizationProjectsPage() {
	const params = useParams();
	const orgId = params.orgId as string;
	const [searchQuery, setSearchQuery] = useState("");

	const organization = organizations[orgId as keyof typeof organizations];
	const orgProjects = projects[orgId as keyof typeof projects] || [];

	const filteredProjects = orgProjects.filter(
		(project) =>
			project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
			project.description.toLowerCase().includes(searchQuery.toLowerCase()),
	);

	const getStatusIcon = (status: string) => {
		switch (status) {
			case "active":
				return <CheckCircle className="h-4 w-4 text-green-600" />;
			case "inactive":
				return <XCircle className="h-4 w-4 text-destructive" />;
			case "paused":
				return <Pause className="h-4 w-4 text-amber-600" />;
			default:
				return <Clock className="h-4 w-4 text-muted-foreground" />;
		}
	};

	const getStatusBadge = (status: string) => {
		switch (status) {
			case "active":
				return <Badge variant="success">Active</Badge>;
			case "inactive":
				return <Badge variant="destructive">Inactive</Badge>;
			case "paused":
				return <Badge variant="warning">Paused</Badge>;
			default:
				return <Badge variant="outline">Unknown</Badge>;
		}
	};

	if (!organization) {
		return <div>Organization not found</div>;
	}

	return (
		<div className="min-h-screen bg-background">
			<div className="container mx-auto max-w-6xl px-4 py-8">
				{/* Header */}
				<div className="mb-8">
					<div className="mb-4">
						<Link href="/api-platform/organizations">
							<Button variant="ghost" size="sm">
								<ArrowLeft className="mr-2 h-4 w-4" />
								Back to Organizations
							</Button>
						</Link>
					</div>
					<h1 className="mb-2 font-bold text-3xl text-foreground">
						{organization.name}
					</h1>
					<p className="mb-6 text-muted-foreground">
						{organization.description}
					</p>

					{/* Stats */}
					<div className="flex flex-wrap gap-8">
						<div className="flex items-center gap-2">
							<div className="rounded-lg bg-muted p-2">
								<Folder className="h-5 w-5 text-muted-foreground" />
							</div>
							<div>
								<div className="font-semibold text-foreground">
									{orgProjects.length}
								</div>
								<div className="text-muted-foreground text-sm">Projects</div>
							</div>
						</div>
						<div className="flex items-center gap-2">
							<div className="rounded-lg bg-muted p-2">
								<Activity className="h-5 w-5 text-muted-foreground" />
							</div>
							<div>
								<div className="font-semibold text-foreground">
									{orgProjects.filter((p) => p.status === "active").length}
								</div>
								<div className="text-muted-foreground text-sm">Active</div>
							</div>
						</div>
					</div>
				</div>

				{/* Search */}
				<div className="mb-8">
					<div className="relative max-w-md">
						<Search className="-translate-y-1/2 absolute top-1/2 left-3 h-4 w-4 transform text-muted-foreground" />
						<Input
							placeholder="Search projects..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="pl-10"
						/>
					</div>
				</div>

				{/* Projects Grid */}
				<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
					{filteredProjects.map((project) => (
						<Card
							key={project.id}
							className="transition-shadow hover:shadow-md"
						>
							<Link
								href={`/api-platform/organizations/${orgId}/projects/${project.id}`}
							>
								<CardHeader className="pb-3">
									<div className="mb-3 flex items-start justify-between">
										<div className="flex items-center gap-3">
											<div className="rounded-lg bg-muted p-2">
												<Folder className="h-5 w-5 text-muted-foreground" />
											</div>
											<div className="min-w-0 flex-1">
												<CardTitle className="truncate text-lg">
													{project.name}
												</CardTitle>
											</div>
										</div>
										<ChevronRight className="h-4 w-4 text-muted-foreground" />
									</div>

									<div className="mb-3 flex items-center gap-2">
										{getStatusIcon(project.status)}
										{getStatusBadge(project.status)}
									</div>

									{/* Progress Bar */}
									<div className="mb-3">
										<div className="mb-1 flex justify-between text-sm">
											<span className="text-muted-foreground">Progress</span>
											<span className="font-medium text-muted-foreground">
												{project.progress}%
											</span>
										</div>
										<div className="h-2 w-full rounded-full bg-muted">
											<div
												className="h-2 rounded-full bg-primary transition-all duration-500"
												style={{ width: `${project.progress}%` }}
											/>
										</div>
									</div>
								</CardHeader>

								<CardContent className="pt-0">
									<CardDescription className="mb-4 line-clamp-2 text-muted-foreground leading-relaxed">
										{project.description}
									</CardDescription>

									<div className="space-y-3 text-sm">
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2 text-muted-foreground">
												<Users className="h-4 w-4" />
												<span>{project.members} members</span>
											</div>
											<div className="flex items-center gap-2 text-muted-foreground">
												<Calendar className="h-4 w-4" />
												<span>{project.createdAt}</span>
											</div>
										</div>
										<div className="flex items-center gap-2 text-muted-foreground">
											<Clock className="h-4 w-4" />
											<span>Updated {project.lastUpdated}</span>
										</div>
									</div>
								</CardContent>
							</Link>
						</Card>
					))}
				</div>

				{filteredProjects.length === 0 && (
					<div className="py-12 text-center">
						<div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-lg bg-muted p-4">
							<Folder className="h-8 w-8 text-muted-foreground" />
						</div>
						<h3 className="mb-2 font-semibold text-foreground text-lg">
							No projects found
						</h3>
						<p className="mx-auto max-w-md text-muted-foreground">
							{searchQuery
								? "Try adjusting your search terms"
								: "This organization doesn't have any projects yet"}
						</p>
					</div>
				)}
			</div>
		</div>
	);
}
