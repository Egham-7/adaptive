"use client";

import { Building2, Check, ChevronDown, Plus } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useOrganizations } from "@/hooks/organizations/use-organizations";

interface OrganizationSelectorProps {
	currentOrganization?: {
		id: string;
		name: string;
	};
}

export function OrganizationSelector({
	currentOrganization,
}: OrganizationSelectorProps) {
	const { orgId } = useParams<{ orgId: string }>();
	const { data: organizations = [], isLoading } = useOrganizations();

	if (isLoading) {
		return (
			<Button
				variant="ghost"
				size="lg"
				disabled
				className="h-12 gap-2 px-4 font-medium text-base"
			>
				<Building2 className="h-5 w-5" />
				Loading...
			</Button>
		);
	}

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button
					variant="ghost"
					size="lg"
					className="h-12 gap-2 px-4 font-medium text-base"
				>
					<Building2 className="h-5 w-5" />
					{currentOrganization?.name
						? `${currentOrganization.name}'s projects`
						: "Select Organization"}
					<ChevronDown className="h-4 w-4" />
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="start" className="w-80">
				<DropdownMenuLabel>Switch Organization</DropdownMenuLabel>
				<DropdownMenuSeparator />

				{organizations.map((org) => (
					<DropdownMenuItem key={org.id} asChild>
						<Link
							href={`/api-platform/organizations/${org.id}`}
							className="flex items-center justify-between"
						>
							<div className="flex items-center gap-2">
								<Building2 className="h-4 w-4" />
								<span className="truncate">{org.name}</span>
							</div>
							{org.id === orgId && <Check className="h-4 w-4" />}
						</Link>
					</DropdownMenuItem>
				))}

				<DropdownMenuSeparator />

				<DropdownMenuItem asChild>
					<Link href="/api-platform/organizations" className="gap-2">
						<Plus className="h-4 w-4" />
						View All Organizations
					</Link>
				</DropdownMenuItem>
			</DropdownMenuContent>
		</DropdownMenu>
	);
}
