import Link from "next/link";
import { Logo } from "@/app/_components/logo";
import { SidebarHeader, useSidebar } from "./ui/sidebar";
import { SocialLogo } from "./ui/social-logo";

export default function CommonSidebarHeader({ href }: { href: string }) {
	const { state } = useSidebar();

	return (
		<SidebarHeader className="flex items-center px-4 py-2">
			<div className="flex items-center gap-2">
				<Link href={href}>
					{state === "collapsed" ? (
						<SocialLogo width={32} height={32} className="h-8 w-8" />
					) : (
						<Logo />
					)}
				</Link>
			</div>
		</SidebarHeader>
	);
}
