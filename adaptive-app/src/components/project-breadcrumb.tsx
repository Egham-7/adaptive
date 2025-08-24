"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import {
	Breadcrumb,
	BreadcrumbItem,
	BreadcrumbLink,
	BreadcrumbList,
	BreadcrumbPage,
	BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/trpc/react";

interface ProjectBreadcrumbProps {
	className?: string;
}

export function ProjectBreadcrumb({ className }: ProjectBreadcrumbProps) {
	const params = useParams<{
		orgId: string;
		projectId: string;
	}>();

	// Safely extract params with proper typing
	const orgId = params?.orgId as string | undefined;
	const projectId = params?.projectId as string | undefined;

	// Only fetch when we have valid IDs
	const { data: organization, isLoading: orgLoading } =
		api.organizations.getById.useQuery(
			{ id: orgId! },
			{ enabled: !!orgId && typeof orgId === "string" && orgId.length > 0 },
		);

	// Fetch project data
	const { data: project, isLoading: projectLoading } =
		api.projects.getById.useQuery(
			{ id: projectId! },
			{
				enabled:
					!!projectId && typeof projectId === "string" && projectId.length > 0,
			},
		);

	if (orgLoading || projectLoading) {
		return (
			<div className={className}>
				<Skeleton className="h-5 w-80" />
			</div>
		);
	}

	if (!organization?.name || !project?.name) {
		return null;
	}

	return (
		<Breadcrumb className={className}>
			<BreadcrumbList>
				<BreadcrumbItem>
					<BreadcrumbLink asChild>
						<Link href={`/api-platform/organizations/${orgId}`}>
							{organization.name}
						</Link>
					</BreadcrumbLink>
				</BreadcrumbItem>
				<BreadcrumbSeparator>
					<span className="mx-1">/</span>
				</BreadcrumbSeparator>
				<BreadcrumbItem>
					<BreadcrumbPage>{project.name}</BreadcrumbPage>
				</BreadcrumbItem>
			</BreadcrumbList>
		</Breadcrumb>
	);
}
