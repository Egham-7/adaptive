"use client";

import { useParams } from "next/navigation";
import {
	Breadcrumb,
	BreadcrumbItem,
	BreadcrumbList,
	BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/trpc/react";

interface OrganizationBreadcrumbProps {
	className?: string;
}

export function OrganizationBreadcrumb({
	className,
}: OrganizationBreadcrumbProps) {
	const { orgId } = useParams<{
		orgId: string;
	}>();

	// Fetch organization data
	const { data: organization, isLoading: orgLoading } =
		api.organizations.getById.useQuery(
			{ id: orgId },
			{ enabled: !!orgId && typeof orgId === 'string' && orgId.length > 0 }
		);

	if (orgLoading) {
		return (
			<div className={className}>
				<Skeleton className="h-5 w-64" />
			</div>
		);
	}

	if (!organization?.name) {
		return null;
	}

	return (
		<Breadcrumb className={className}>
			<BreadcrumbList>
				<BreadcrumbItem>
					<BreadcrumbPage>{organization.name} /</BreadcrumbPage>
				</BreadcrumbItem>
			</BreadcrumbList>
		</Breadcrumb>
	);
}
