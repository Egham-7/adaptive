// src/app/_components/api-platform/sidebar-nav-footer.tsx

import { useUser } from "@clerk/nextjs";
import {
	SidebarFooter,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarSeparator,
} from "@/components/ui/sidebar";
import { SupportButton } from "@/components/ui/support-button";
import { ModeToggle } from "../mode-toggle";
import { ApiNavUser } from "./api-nav-user";

export function ApiSidebarNavFooter() {
	const { user } = useUser();

	if (!user) {
		return null;
	}

	return (
		<SidebarFooter>
			<SidebarSeparator />
			<SidebarMenu className="flex-row items-center justify-between p-2">
				<SidebarMenuItem>
					<SidebarMenuButton asChild>
						<ApiNavUser
							user={{
								name: user.firstName || "Unknown User",
								email: user.emailAddresses?.[0]?.emailAddress || "No Email",
								avatar: user.imageUrl || "/default-avatar.png",
							}}
						/>
					</SidebarMenuButton>
				</SidebarMenuItem>
				<div className="flex items-center gap-1">
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<SupportButton variant="ghost" size="icon" showText={false} />
						</SidebarMenuButton>
					</SidebarMenuItem>
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<ModeToggle />
						</SidebarMenuButton>
					</SidebarMenuItem>
				</div>
			</SidebarMenu>
		</SidebarFooter>
	);
}
