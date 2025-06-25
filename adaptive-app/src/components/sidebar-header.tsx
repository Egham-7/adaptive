import { Logo } from "@/app/_components/logo";
import Link from "next/link";
import { SidebarHeader } from "./ui/sidebar";

export default function CommonSidebarHeader({ href }: { href: string }) {
	return (
		<SidebarHeader className="flex items-center px-4 py-2">
			<div className="flex items-center gap-2">
				<Link href="/">
					<Logo />
				</Link>
				<Link className="font-bold text-xl" href={href}>
					Adaptive
				</Link>
			</div>
		</SidebarHeader>
	);
}
