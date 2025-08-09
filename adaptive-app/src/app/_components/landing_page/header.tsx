"use client";

import Link from "next/link";
import { Logo } from "../logo";
import { DesktopNavigation } from "./header/desktop-navigation";
import { UserActions } from "./header/user-actions";
import { MobileHeader } from "./mobile-header";

export default function Header() {
	return (
		<>
			{/* Mobile Header */}
			<MobileHeader />

			{/* Desktop Header */}
			<header className="hidden lg:block">
				<nav className="relative z-20 w-full border-b border-dashed bg-background dark:bg-background/50">
					<div className="m-auto max-w-5xl px-6">
						<div className="flex items-center justify-between py-3 lg:py-4">
							{/* Logo */}
							<Link
								href="/"
								aria-label="Adaptive AI Home"
								className="flex items-center space-x-2"
							>
								<Logo />
							</Link>

							{/* Desktop Navigation */}
							<DesktopNavigation />
							<UserActions />
						</div>
					</div>
				</nav>
			</header>
		</>
	);
}
