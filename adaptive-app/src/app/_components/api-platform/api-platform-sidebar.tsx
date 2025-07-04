"use client";
import { motion } from "framer-motion";
import { LayoutDashboard, LogOut, Settings, UserCog } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "@/components/ui/sidebar";
import { ModeToggle } from "../mode-toggle";

export function APIPlatformSidebar() {
	const links = [
		{
			label: "Dashboard",
			href: "#",
			icon: (
				<LayoutDashboard className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
		{
			label: "Profile",
			href: "#",
			icon: (
				<UserCog className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
		{
			label: "Settings",
			href: "#",
			icon: (
				<Settings className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
		{
			label: "Logout",
			href: "#",
			icon: (
				<LogOut className="h-5 w-5 flex-shrink-0 text-neutral-700 dark:text-neutral-200" />
			),
		},
	];
	const [open, setOpen] = useState(false);
	return (
		<Sidebar open={open} setOpen={setOpen}>
			<SidebarBody className="justify-between gap-10">
				<div className="flex flex-1 flex-col overflow-y-auto overflow-x-hidden">
					{open ? <Logo /> : <LogoIcon />}
					<div className="mt-8 flex flex-col gap-2">
						{links.map((link) => (
							<SidebarLink key={link.label} link={link} />
						))}
					</div>
				</div>
				<div className="flex items-center justify-between gap-2">
					<SidebarLink
						link={{
							label: "Manu Arora",
							href: "#",
							icon: (
								<Image
									src="https://assets.aceternity.com/manu.png"
									className="h-7 w-7 flex-shrink-0 rounded-full"
									width={50}
									height={50}
									alt="Avatar"
								/>
							),
						}}
					/>
					<ModeToggle />
				</div>
			</SidebarBody>
		</Sidebar>
	);
}

export const Logo = () => {
	return (
		<Link
			href="#"
			className="relative z-20 flex items-center space-x-2 py-1 font-normal text-black text-sm"
		>
			<div className="h-5 w-6 flex-shrink-0 rounded-tl-lg rounded-tr-sm rounded-br-lg rounded-bl-sm bg-black dark:bg-white" />
			<motion.span
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				className="whitespace-pre font-medium text-black dark:text-white"
			>
				Acet Labs
			</motion.span>
		</Link>
	);
};

export const LogoIcon = () => {
	return (
		<Link
			href="#"
			className="relative z-20 flex items-center space-x-2 py-1 font-normal text-black text-sm"
		>
			<div className="h-5 w-6 flex-shrink-0 rounded-tl-lg rounded-tr-sm rounded-br-lg rounded-bl-sm bg-black dark:bg-white" />
		</Link>
	);
};
