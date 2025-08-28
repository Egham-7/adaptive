// src/app/_components/api-platform/sidebar-nav-footer.tsx

import { useUser } from "@clerk/nextjs";
import { useId } from "react";
import { DocsButton } from "@/components/ui/docs-button";
import { LegalButton } from "@/components/ui/legal-button";
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
	const docsId = useId();
	const supportId = useId();
	const legalId = useId();

	if (!user) {
		return null;
	}

	return (
		<SidebarFooter>
			<SidebarSeparator />
			<SidebarMenu className="space-y-2 p-2">
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
				<div className="flex flex-wrap items-center justify-center gap-1">
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<div id={docsId}>
								<DocsButton variant="ghost" size="icon" showText={false} />
							</div>
						</SidebarMenuButton>
					</SidebarMenuItem>
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<div id={supportId}>
								<SupportButton variant="ghost" size="icon" showText={false} />
							</div>
						</SidebarMenuButton>
					</SidebarMenuItem>
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<div id={legalId}>
								<LegalButton variant="ghost" size="icon" showText={false} />
							</div>
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
