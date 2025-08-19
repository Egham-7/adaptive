import type { ReactNode } from "react";

import { OrganizationSidebar } from "@/app/_components/api-platform/organizations/organization-sidebar";
import { OrganizationBreadcrumb } from "@/components/organization-breadcrumb";
import {
	SidebarInset,
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";

export default function OrganizationLayout({
	children,
}: {
	children: ReactNode;
}) {
	return (
		<SidebarProvider>
			<OrganizationSidebar />
			<SidebarInset>
				<div className="flex h-16 shrink-0 items-center gap-2 px-4">
					<SidebarTrigger className="-ml-1" />
				</div>
				<div className="px-4 pb-2">
					<OrganizationBreadcrumb />
				</div>
				<main className="flex flex-1 flex-col gap-4 p-4 pt-0">{children}</main>
			</SidebarInset>
		</SidebarProvider>
	);
}
