"use client";

import { SignedIn, SignedOut } from "@clerk/nextjs";
import { Menu, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Logo } from "../logo";
import { ModeToggle } from "../mode-toggle";

const menuItems = [
	{ name: "Features", href: "/features" },
	{ name: "Solution", href: "/solution" },
	{ name: "Pricing", href: "/pricing" },
	{ name: "About", href: "/about" },
	{ name: "Support", href: "/support" },
];

export function MobileHeader() {
	const [isOpen, setIsOpen] = useState(false);

	const closeMenu = () => setIsOpen(false);

	return (
		<>
			{/* Fixed Mobile Header Bar */}
			<header className="fixed top-0 right-0 left-0 z-50 border-b bg-background lg:hidden">
				<div className="flex items-center justify-between px-4">
					<Link href="/" onClick={closeMenu}>
						<Logo imageWidth={80} imageHeight={67} textSize="base" />
					</Link>

					<Button
						variant="ghost"
						size="icon"
						onClick={() => setIsOpen(!isOpen)}
						aria-label={isOpen ? "Close menu" : "Open menu"}
					>
						{isOpen ? <X size={24} /> : <Menu size={24} />}
					</Button>
				</div>
			</header>

			{/* Mobile Menu Overlay */}
			{isOpen && (
				<div className="fixed inset-0 z-40 lg:hidden">
					<button
						type="button"
						className="absolute inset-0 cursor-default bg-background/90 backdrop-blur-sm"
						onClick={closeMenu}
						aria-label="Close menu"
					/>

					<div className="relative flex min-h-full flex-col items-center justify-center space-y-8 p-8">
						{/* Navigation Links */}
						<nav className="space-y-6 text-center">
							{menuItems.map((item) => (
								<Link
									key={item.name}
									href={item.href}
									onClick={closeMenu}
									className="block font-medium text-lg text-muted-foreground transition-colors hover:text-foreground"
								>
									{item.name}
								</Link>
							))}
						</nav>

						{/* Documentation Link */}
						<a
							href="https://docs.llmadaptive.uk/"
							target="_blank"
							rel="noopener noreferrer"
							onClick={closeMenu}
							className="text-muted-foreground transition-colors hover:text-foreground"
						>
							Documentation
						</a>

						{/* Auth Buttons */}
						<div className="flex w-full max-w-xs flex-col items-center space-y-4">
							<SignedOut>
								<Link href="/sign-in?redirect_url=/chat-platform">
									<Button
										variant="outline"
										size="lg"
										className="w-full"
										onClick={closeMenu}
									>
										Sign In - Chat App
									</Button>
								</Link>

								<Link href="/sign-in?redirect_url=/api-platform/organizations">
									<Button
										variant="outline"
										size="lg"
										className="w-full"
										onClick={closeMenu}
									>
										Sign In - API Platform
									</Button>
								</Link>

								<div className="my-2 h-px w-full bg-border" />

								<Link href="/sign-up?redirect_url=/chat-platform">
									<Button size="lg" className="w-full" onClick={closeMenu}>
										Get Started - Chat App
									</Button>
								</Link>

								<Link href="/sign-up?redirect_url=/api-platform/organizations">
									<Button size="lg" className="w-full" onClick={closeMenu}>
										Get Started - API Platform
									</Button>
								</Link>
							</SignedOut>

							<SignedIn>
								<Link href="/chat-platform" onClick={closeMenu}>
									<Button variant="outline" size="lg" className="w-full">
										Chat Platform
									</Button>
								</Link>

								<Link href="/api-platform/organizations" onClick={closeMenu}>
									<Button variant="outline" size="lg" className="w-full">
										API Platform
									</Button>
								</Link>
							</SignedIn>
						</div>

						{/* Theme Toggle */}
						<ModeToggle />
					</div>
				</div>
			)}
		</>
	);
}
