"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { MdDashboard, MdKey } from "react-icons/md";
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

export function ProjectSidebar() {
	const { orgId, projectId } = useParams();

	const links = [
		{
			label: "Dashboard",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}`,
			icon: (
				<MdDashboard className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
		{
			label: "API Keys",
			href: `/api-platform/organizations/${orgId}/projects/${projectId}/api-keys`,
			icon: (
				<MdKey className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
	];
	return (
		<Sidebar className="h-screen">
			<div className="flex items-center justify-between space-x-2 p-2">
				<CommonSidebarHeader
					href={`/api-platform/organizations/${orgId}/projects/${projectId}`}
				/>
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
