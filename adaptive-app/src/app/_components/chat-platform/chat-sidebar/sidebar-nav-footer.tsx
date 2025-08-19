// src/components/chat/chat-sidebar/SidebarNavFooter.tsx

import { useUser } from "@clerk/nextjs";
import { LegalButton } from "@/components/ui/legal-button";
import {
	SidebarFooter,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarSeparator,
} from "@/components/ui/sidebar";
import { SupportButton } from "@/components/ui/support-button";
import { ModeToggle } from "../../mode-toggle";
import { NavUser } from "./nav-user";

export function SidebarNavFooter() {
	const { user } = useUser(); // Fetch user data from Clerk's useUser hook

	if (!user) {
		return null;
	}

	return (
		<SidebarFooter>
			<SidebarSeparator />
			<SidebarMenu className="flex-row items-center justify-between p-2">
				<SidebarMenuItem>
					<SidebarMenuButton asChild>
						<NavUser
							user={{
								name: user.firstName || "Unknown User", // Provide a default value
								email: user.emailAddresses?.[0]?.emailAddress || "No Email", // Handle array and provide default
								avatar: user.imageUrl || "/default-avatar.png", // Provide a default avatar
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
							<LegalButton variant="ghost" size="icon" showText={false} />
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
