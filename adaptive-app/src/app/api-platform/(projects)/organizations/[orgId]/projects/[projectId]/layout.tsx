"use client";

import type { ReactNode } from "react";
import { ProjectSidebar } from "@/app/_components/api-platform/organizations/projects/project-sidebar";
import { ProjectBreadcrumb } from "@/components/project-breadcrumb";
import { NewUserTour } from "@/components/ui/new-user-tour";
import {
	SidebarInset,
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";

export default function ProjectLayout({ children }: { children: ReactNode }) {
	return (
		<>
			<SidebarProvider>
				<ProjectSidebar />
				<SidebarInset>
					<div className="flex h-16 shrink-0 items-center gap-2 px-4">
						<SidebarTrigger className="-ml-1" />
					</div>
					<div className="px-4 pb-2">
						<ProjectBreadcrumb />
					</div>
					<main className="flex flex-1 flex-col gap-4 p-4 pt-0">
						{children}
					</main>
				</SidebarInset>
			</SidebarProvider>
			<NewUserTour />
		</>
	);
}
