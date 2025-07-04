import { APIPlatformSidebar } from "../_components/api-platform/api-platform-sidebar";
export default async function APIPlatformLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<div className="flex min-h-screen w-full bg-background">
			<APIPlatformSidebar />
			<main className="w-full flex-1 overflow-hidden px-10 py-2 pt-10">
				{children}
			</main>
		</div>
	);
}
