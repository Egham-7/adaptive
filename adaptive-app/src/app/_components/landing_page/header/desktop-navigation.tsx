"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { GitHubStarsButton } from "@/components/animate-ui/buttons/github-stars";
import { iconMenuItems, menuItems } from "./navigation-items";

export function DesktopNavigation() {
	const pathname = usePathname();

	return (
		<div className="hidden items-center gap-6 lg:flex">
			{/* Main navigation */}
			<nav className="flex items-center gap-1 text-sm">
				{menuItems.map((item) => {
					const isActive =
						item.href === "/"
							? pathname === item.href
							: pathname.startsWith(item.href);
					return (
						<div key={item.name} className="relative">
							{item.external ? (
								<a
									href={item.href}
									target="_blank"
									rel="noopener noreferrer"
									className="relative cursor-pointer rounded-full px-4 py-2 font-medium text-muted-foreground text-sm transition-colors hover:text-accent-foreground"
								>
									{item.name}
								</a>
							) : (
								<Link
									href={item.href}
									className={`relative cursor-pointer rounded-full px-4 py-2 font-medium text-sm transition-colors ${
										isActive
											? "text-primary"
											: "text-muted-foreground hover:text-accent-foreground"
									}`}
									aria-current={isActive ? "page" : undefined}
								>
									{item.name}
									{isActive && (
										<motion.div
											layoutId="desktop-nav-lamp"
											className="-z-10 absolute inset-0 w-full rounded-full bg-primary/5"
											initial={false}
											transition={{
												type: "spring",
												stiffness: 300,
												damping: 30,
											}}
										>
											<div className="-top-2 -translate-x-1/2 absolute left-1/2 h-1 w-8 rounded-t-full bg-primary">
												<div className="-top-2 -left-2 absolute h-6 w-12 rounded-full bg-primary/20 blur-md" />
												<div className="-top-1 absolute h-6 w-8 rounded-full bg-primary/20 blur-md" />
												<div className="absolute top-0 left-2 h-4 w-4 rounded-full bg-primary/20 blur-sm" />
											</div>
										</motion.div>
									)}
								</Link>
							)}
						</div>
					);
				})}
			</nav>

			{/* Separator */}
			<div className="h-4 w-px bg-border" />

			{/* External links */}
			<nav className="flex items-center gap-3" aria-label="External links">
				{iconMenuItems.map((item) => {
					const Icon = item.icon;
					return (
						<a
							key={item.name}
							href={item.href}
							target="_blank"
							rel="noopener noreferrer"
							className="flex items-center gap-1 text-muted-foreground duration-150 hover:text-accent-foreground"
							aria-label={`Visit ${item.name}`}
						>
							<Icon size={16} aria-hidden={true} />
							<span className="text-sm">{item.name}</span>
						</a>
					);
				})}

				{/* GitHub Stars Button */}
				<GitHubStarsButton
					username="Egham-7"
					repo="adaptive"
					formatted={true}
					className="h-8 text-xs"
				/>
			</nav>
		</div>
	);
}
