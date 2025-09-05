import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuItem,
	SidebarRail,
	SidebarSeparator,
	SidebarTrigger,
} from "@/components/ui/sidebar";
import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
	return (
		<div className="flex min-h-screen w-full">
			<Sidebar className="h-screen">
				{/* Header with logo and trigger */}
				<div className="flex items-center justify-between space-x-2 p-2">
					<SidebarHeader className="flex items-center px-4 py-2">
						<div className="flex items-center gap-2">
							<Skeleton className="h-8 w-8 rounded" />
							<Skeleton className="h-6 w-20" />
						</div>
					</SidebarHeader>
					<SidebarTrigger />
				</div>
				<SidebarSeparator />

				{/* Actions section - New Chat button and search */}
				<div className="p-3">
					<Skeleton className="h-10 w-full rounded-md" />
				</div>
				<div className="px-3 pb-2">
					<Skeleton className="h-10 w-full rounded-md" />
				</div>

				{/* Conversation list */}
				<SidebarContent className="px-1">
					<div className="space-y-4">
						{/* Pinned section */}
						<div className="space-y-2">
							<div className="flex items-center gap-1 px-2">
								<Skeleton className="h-3 w-3" />
								<Skeleton className="h-4 w-12" />
							</div>
							{Array.from({ length: 2 }).map((_, i) => (
								<Skeleton
									key={`pinned-${i}`} // This will now be ignored by Biome
									className="mx-2 h-12 rounded-md"
								/>
							))}
						</div>

						{/* Today section */}
						<div className="space-y-2">
							<Skeleton className="mx-2 h-4 w-12" />
							{Array.from({ length: 3 }).map((_, i) => (
								<Skeleton
									key={`today-${i}`} // This will now be ignored by Biome
									className="mx-2 h-12 rounded-md"
								/>
							))}
						</div>

						{/* Last 30 Days section */}
						<div className="space-y-2">
							<Skeleton className="mx-2 h-4 w-20" />
							{Array.from({ length: 4 }).map((_, i) => (
								<Skeleton
									key={`last30-${i}`} // This will now be ignored by Biome
									className="mx-2 h-12 rounded-md"
								/>
							))}
						</div>
					</div>
				</SidebarContent>

				{/* Footer with user button and theme toggle */}
				<SidebarFooter>
					<SidebarSeparator />
					<SidebarMenu className="flex-row items-center justify-between p-2">
						<SidebarMenuItem>
							<Skeleton className="h-8 w-8 rounded-full" />
						</SidebarMenuItem>
						<SidebarMenuItem>
							<Skeleton className="h-8 w-8 rounded-md" />
						</SidebarMenuItem>
					</SidebarMenu>
				</SidebarFooter>

				<SidebarRail />
			</Sidebar>

			{/* Main content area */}
			<main className="w-full flex-1 overflow-hidden px-10 py-2 pt-10">
				<Skeleton className="h-8 max-w-md rounded bg-muted" />
			</main>
		</div>
	);
}
