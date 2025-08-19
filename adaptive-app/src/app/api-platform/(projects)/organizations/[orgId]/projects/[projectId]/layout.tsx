"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import type { ReactNode } from "react";
import { ProjectSidebar } from "@/app/_components/api-platform/organizations/projects/project-sidebar";
import { ProjectBreadcrumb } from "@/components/project-breadcrumb";
import { Button } from "@/components/ui/button";
import {
	SidebarInset,
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";

export default function ProjectLayout({ children }: { children: ReactNode }) {
	const { orgId } = useParams<{ orgId: string }>();
	return (
		<SidebarProvider>
			<ProjectSidebar />
			<SidebarInset>
				<div className="flex h-16 shrink-0 items-center gap-2 px-4">
					<SidebarTrigger className="-ml-1" />
				</div>
				<div className="px-4 pb-2">
					<ProjectBreadcrumb />
				</div>
				<div className="px-4 pb-4">
					<Link href={`/api-platform/organizations/${orgId}`}>
						<Button variant="ghost" size="sm">
							<ArrowLeft className="mr-2 h-4 w-4" />
							Back to Projects
						</Button>
					</Link>
				</div>
				<main className="flex flex-1 flex-col gap-4 p-4 pt-0">{children}</main>
			</SidebarInset>
		</SidebarProvider>
	);
}
