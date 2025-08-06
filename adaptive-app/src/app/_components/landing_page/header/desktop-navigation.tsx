"use client";

import Link from "next/link";
import { GitHubStarsButton } from "@/components/animate-ui/buttons/github-stars";
import { iconMenuItems, menuItems } from "./navigation-items";

export function DesktopNavigation() {
	return (
		<div className="hidden items-center gap-6 lg:flex">
			{/* Main navigation */}
			<nav className="flex items-center gap-6 text-sm">
				{menuItems.map((item) => (
					<div key={item.name}>
						{item.external ? (
							<a
								href={item.href}
								target="_blank"
								rel="noopener noreferrer"
								className="text-muted-foreground duration-150 hover:text-accent-foreground"
								role="menuitem"
							>
								{item.name}
							</a>
						) : (
							<Link
								href={item.href}
								className="text-muted-foreground duration-150 hover:text-accent-foreground"
								role="menuitem"
							>
								{item.name}
							</Link>
						)}
					</div>
				))}
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
