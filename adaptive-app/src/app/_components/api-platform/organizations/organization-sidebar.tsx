"use client";
import type { ReactNode } from "react";

import Link from "next/link";
import { useParams } from "next/navigation";
import {
	MdCode,
	MdCreditCard,
	MdDashboard,
	MdFolder,
	MdKey,
	MdPlayArrow,
} from "react-icons/md";
import { ApiSidebarNavFooter } from "@/app/_components/api-platform/sidebar-nav-footer";
import CommonSidebarHeader from "@/components/sidebar-header";
import {
	Sidebar,
	SidebarContent,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarRail,
	SidebarSeparator,
	SidebarTrigger,
} from "@/components/ui/sidebar";

interface NavLink {
	label: string;
	href: string;
	icon: ReactNode;
}

export function OrganizationSidebar() {
	const { orgId, projectId } = useParams<{
		orgId: string;
		projectId?: string;
	}>();

	// If we're on a project page, show project navigation
	const isProjectPage = typeof projectId === "string";

	const organizationLinks: NavLink[] = [
		{
			label: "Projects",
			href: `/api-platform/organizations/${orgId}`,
			icon: <MdFolder className="h-5 w-5" />,
		},
		{
			label: "Credits & Billing",
			href: `/api-platform/organizations/${orgId}?tab=credits`,
			icon: <MdCreditCard className="h-5 w-5" />,
		},
	];

	const projectLinks: NavLink[] = [
		{
			label: "Dashboard",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}`,
			icon: <MdDashboard className="h-5 w-5" />,
		},
		{
			label: "Quickstart",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}/quickstart`,
			icon: <MdPlayArrow className="h-5 w-5" />,
		},
		{
			label: "Examples",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}/examples`,
			icon: <MdCode className="h-5 w-5" />,
		},
		{
			label: "API Keys",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}/api-keys`,
			icon: <MdKey className="h-5 w-5" />,
		},
	];

	const links = isProjectPage ? projectLinks : organizationLinks;

	return (
		<Sidebar className="h-screen">
			<div className="flex items-center justify-between space-x-2 p-2">
				<CommonSidebarHeader href={`/api-platform/organizations/${orgId}`} />
				<SidebarTrigger />
			</div>
			<SidebarSeparator />

			<SidebarContent>
				<SidebarMenu>
					{links.map((link) => (
						<SidebarMenuItem key={link.label}>
							<SidebarMenuButton asChild>
								<Link href={link.href}>
									{link.icon}
									<span>{link.label}</span>
								</Link>
							</SidebarMenuButton>
						</SidebarMenuItem>
					))}
				</SidebarMenu>
			</SidebarContent>

			<ApiSidebarNavFooter />
			<SidebarRail />
		</Sidebar>
	);
}
