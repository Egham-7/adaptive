import { APIPlatformSidebar } from "../_components/api-platform/api-platform-sidebar";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";

export default async function APIPlatformLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<SidebarProvider>
			<APIPlatformSidebar />
			<SidebarInset>
				<div className="flex h-16 shrink-0 items-center gap-2 px-4">
					<SidebarTrigger className="-ml-1" />
				</div>
				<main className="flex flex-1 flex-col gap-4 p-4 pt-0">
					{children}
				</main>
			</SidebarInset>
		</SidebarProvider>
	);
}
