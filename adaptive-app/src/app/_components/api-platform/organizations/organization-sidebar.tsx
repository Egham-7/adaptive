"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import type { ReactNode } from "react";
import { MdCreditCard, MdFolder } from "react-icons/md";
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

export function OrganizationSidebar() {
	const { orgId } = useParams<{
		orgId: string;
	}>();

	// Organization-level navigation - always visible
	const organizationLinks: NavLink[] = [
		{
			label: "Projects",
			href: `/api-platform/organizations/${orgId}`,
			icon: <MdFolder className="h-5 w-5" />,
		},
		{
			label: "Credits & Billing",
			href: `/api-platform/organizations/${orgId}/credits`,
			icon: <MdCreditCard className="h-5 w-5" />,
		},
	];

	// No project-level navigation in organization sidebar

	return (
		<Sidebar collapsible="icon" className="h-screen">
			<div className="flex items-center justify-center space-x-2 p-2">
				<CommonSidebarHeader href={`/api-platform/organizations/${orgId}`} />
			</div>
			<SidebarSeparator />

			<SidebarContent>
				<SidebarMenu>
					{/* Organization-level navigation only */}
					{organizationLinks.map((link) => (
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
