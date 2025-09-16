import type { ReactNode } from "react";

import { OrganizationSidebar } from "@/app/_components/api-platform/organizations/organization-sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

export default function OrganizationLayout({
	children,
}: {
	children: ReactNode;
}) {
	return (
		<SidebarProvider>
			<OrganizationSidebar />
			<SidebarInset>
				<main className="flex flex-1 flex-col gap-4 p-4">{children}</main>
			</SidebarInset>
		</SidebarProvider>
	);
}
