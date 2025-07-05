"use client";
import { motion } from "framer-motion";
import { LayoutDashboard, LogOut, Settings, UserCog } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
} from "@/components/ui/sidebar";
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
	return (
		<Sidebar>
			<SidebarHeader>
				<Logo />
			</SidebarHeader>
			<SidebarContent>
				<SidebarMenu>
					{links.map((link) => (
						<SidebarMenuItem key={link.label}>
							<SidebarMenuButton asChild>
								<Link href={link.href}>
									{link.icon}
									<span>{link.label}</span>
								</Link>
							</SidebarMenuButton>
						</SidebarMenuItem>
					))}
				</SidebarMenu>
			</SidebarContent>
			<SidebarFooter>
				<SidebarMenu>
					<SidebarMenuItem>
						<SidebarMenuButton asChild>
							<Link href="#" className="flex items-center gap-2">
								<Image
									src="https://assets.aceternity.com/manu.png"
									className="h-7 w-7 flex-shrink-0 rounded-full"
									width={50}
									height={50}
									alt="Avatar"
								/>
								<span>Manu Arora</span>
							</Link>
						</SidebarMenuButton>
					</SidebarMenuItem>
				</SidebarMenu>
				<ModeToggle />
			</SidebarFooter>
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
