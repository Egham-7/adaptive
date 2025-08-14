"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { MdCreditCard, MdFolder, MdSettings } from "react-icons/md";
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

export function OrganizationSidebar() {
	const { orgId } = useParams();

	const links = [
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
		{
			label: "Settings",
			href: `/api-platform/organizations/${orgId}/settings`,
			icon: <MdSettings className="h-5 w-5" />,
		},
	];

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
