"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import type { ReactNode } from "react";
import { MdCode, MdDashboard, MdKey, MdPlayArrow } from "react-icons/md";
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
} from "@/components/ui/sidebar";

interface NavLink {
	label: string;
	href: string;
	icon: ReactNode;
}

export function ProjectSidebar() {
	const { orgId, projectId } = useParams<{
		orgId: string;
		projectId: string;
	}>();

	// Project-level navigation
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

	return (
		<Sidebar collapsible="icon" className="h-screen">
			<div className="flex items-center justify-center space-x-2 p-2">
				<CommonSidebarHeader href={`/api-platform/organizations/${orgId}`} />
			</div>
			<SidebarSeparator />

			<SidebarContent>
				<SidebarMenu>
					{/* Project-level navigation */}
					{projectLinks.map((link) => (
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
