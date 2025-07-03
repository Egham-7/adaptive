"use client";

import { SignedIn, SignedOut, SignInButton, SignUpButton } from "@clerk/nextjs";
import { Book, ChevronDown, Github, Menu, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Logo } from "../logo";
import { ModeToggle } from "../mode-toggle";

const menuItems = [
	{ name: "Features", href: "#features" },
	{ name: "Solution", href: "#solution" },
	{ name: "Pricing", href: "#pricing" },
	{ name: "About", href: "#about" },
];

type MenuItem = {
	name: string;
	href: string;
	external?: boolean;
};

const iconMenuItems = [
	{
		name: "Docs",
		href: "https://docs.adaptive.dev",
		icon: Book,
		external: true,
	},
	{
		name: "GitHub",
		href: "https://github.com/Egham-7/adaptive",
		icon: Github,
		external: true,
	},
];

export default function Header() {
	const [menuState, setMenuState] = useState(false);

	return (
		<header>
			<nav
				data-state={menuState && "active"}
				className="fixed z-20 w-full border-b border-dashed bg-background backdrop-blur-sm md:relative dark:bg-background/50 lg:dark:bg-transparent"
			>
				<div className="m-auto max-w-5xl px-6">
					<div className="flex flex-wrap items-center justify-between gap-6 py-3 lg:gap-0 lg:py-4">
						{/* Logo section */}
						<div className="flex w-full justify-between lg:w-auto">
							<Link
								href="/"
								aria-label="home"
								className="flex items-center space-x-2"
							>
								<Logo />
							</Link>
							<button
								type="button"
								onClick={() => setMenuState(!menuState)}
								aria-label={menuState === true ? "Close Menu" : "Open Menu"}
								className="-m-2.5 -mr-4 relative z-20 block cursor-pointer p-2.5 lg:hidden"
							>
								<Menu className="m-auto size-6 in-data-[state=active]:rotate-180 in-data-[state=active]:scale-0 in-data-[state=active]:opacity-0 duration-200" />
								<X className="-rotate-180 absolute inset-0 m-auto size-6 in-data-[state=active]:rotate-0 in-data-[state=active]:scale-100 scale-0 in-data-[state=active]:opacity-100 opacity-0 duration-200" />
							</button>
						</div>

						{/* Menu section - hidden on mobile, centered on desktop */}
						<div className="hidden lg:block">
							<ul className="flex gap-8 text-sm">
								{menuItems.map((item) => (
									<li key={item.name}>
										{(item as MenuItem).external ? (
											<a
												href={item.href}
												target="_blank"
												rel="noopener noreferrer"
												className="text-muted-foreground duration-150 hover:text-accent-foreground"
											>
												<span>{item.name}</span>
											</a>
										) : (
											<Link
												href={item.href}
												className="text-muted-foreground duration-150 hover:text-accent-foreground"
											>
												<span>{item.name}</span>
											</Link>
										)}
									</li>
								))}
							</ul>
						</div>

						{/* Icon links - docs and GitHub */}
						<div className="hidden lg:flex lg:items-center lg:gap-3">
							{iconMenuItems.map((item) => {
								const Icon = item.icon;
								return (
									<a
										key={item.name}
										href={item.href}
										target="_blank"
										rel="noopener noreferrer"
										className="text-muted-foreground duration-150 hover:text-accent-foreground"
										title={item.name}
									>
										<Icon className="h-5 w-5" />
									</a>
								);
							})}
						</div>

						{/* Buttons section - hidden on mobile, shown on desktop */}
						<div className="hidden lg:flex lg:items-center lg:gap-3">
							<SignedOut>
								<DropdownMenu>
									<DropdownMenuTrigger asChild>
										<Button
											variant="ghost"
											size="sm"
											className="flex items-center gap-1 font-medium hover:bg-primary/50 hover:text-primary-foreground"
										>
											Login
											<ChevronDown className="h-4 w-4" />
										</Button>
									</DropdownMenuTrigger>
									<DropdownMenuContent align="end">
										<DropdownMenuItem>
											<SignInButton signUpForceRedirectUrl="/chat-platform">
												Chatbot App
											</SignInButton>
										</DropdownMenuItem>
										<DropdownMenuItem>
											<SignInButton signUpForceRedirectUrl="/api-platform">
												API Platform
											</SignInButton>
										</DropdownMenuItem>
									</DropdownMenuContent>
								</DropdownMenu>

								<DropdownMenu>
									<DropdownMenuTrigger asChild>
										<Button
											size="sm"
											className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
										>
											Get Started
											<ChevronDown className="ml-1 h-4 w-4" />
										</Button>
									</DropdownMenuTrigger>
									<DropdownMenuContent align="end">
										<DropdownMenuItem>
											<SignUpButton signInForceRedirectUrl="/chat-platform">
												Chatbot App
											</SignUpButton>
										</DropdownMenuItem>
										<DropdownMenuItem>
											<SignUpButton signInForceRedirectUrl="/api-platform">
												API Platform
											</SignUpButton>
										</DropdownMenuItem>
									</DropdownMenuContent>
								</DropdownMenu>
							</SignedOut>

							<SignedIn>
								<DropdownMenu>
									<DropdownMenuTrigger asChild>
										<Button
											variant="ghost"
											size="sm"
											className="flex items-center gap-1 font-medium hover:bg-primary/50 hover:text-primary-foreground"
										>
											My Account
											<ChevronDown className="h-4 w-4" />
										</Button>
									</DropdownMenuTrigger>
									<DropdownMenuContent align="end">
										<DropdownMenuItem asChild>
											<Link href="/chat-platform">Chatbot App</Link>
										</DropdownMenuItem>
										<DropdownMenuItem asChild>
											<Link href="/api-platform">API Platform</Link>
										</DropdownMenuItem>
									</DropdownMenuContent>
								</DropdownMenu>
							</SignedIn>

							<ModeToggle />
						</div>

						{/* Mobile menu - keep existing mobile menu structure */}
						<div className="mb-6 in-data-[state=active]:block hidden w-full flex-wrap items-center justify-center space-y-8 rounded-3xl border bg-background p-6 shadow-2xl shadow-color-shadow-color/[var(--shadow-opacity)] md:flex-nowrap lg:hidden">
							<div>
								<ul className="space-y-6 text-base lg:flex lg:justify-center lg:gap-8 lg:space-y-0 lg:text-sm">
									{menuItems.map((item) => (
										<li key={item.name}>
											{(item as MenuItem).external ? (
												<a
													href={item.href}
													target="_blank"
													rel="noopener noreferrer"
													className="block text-muted-foreground duration-150 hover:text-accent-foreground"
												>
													<span>{item.name}</span>
												</a>
											) : (
												<Link
													href={item.href}
													className="block text-muted-foreground duration-150 hover:text-accent-foreground"
												>
													<span>{item.name}</span>
												</Link>
											)}
										</li>
									))}
								</ul>

								{/* Icon links for mobile */}
								<div className="flex justify-center gap-6 pt-4">
									{iconMenuItems.map((item) => {
										const Icon = item.icon;
										return (
											<a
												key={item.name}
												href={item.href}
												target="_blank"
												rel="noopener noreferrer"
												className="flex items-center gap-2 text-muted-foreground duration-150 hover:text-accent-foreground"
											>
												<Icon className="h-5 w-5" />
												<span>{item.name}</span>
											</a>
										);
									})}
								</div>
							</div>

							<div className="flex w-full flex-col space-y-3 sm:flex-row sm:gap-3 sm:space-y-0 md:w-fit lg:border-l lg:pl-6">
								<SignedOut>
									<DropdownMenu>
										<DropdownMenuTrigger asChild>
											<Button
												variant="ghost"
												size="sm"
												className="flex items-center gap-1 font-medium hover:bg-primary/50 hover:text-primary-foreground"
											>
												Login
												<ChevronDown className="h-4 w-4" />
											</Button>
										</DropdownMenuTrigger>
										<DropdownMenuContent align="end">
											<DropdownMenuItem>
												<SignInButton signUpForceRedirectUrl="/chat-platform">
													Chatbot App
												</SignInButton>
											</DropdownMenuItem>
											<DropdownMenuItem>
												<SignInButton signUpForceRedirectUrl="/api-platform">
													API Platform
												</SignInButton>
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>

									<DropdownMenu>
										<DropdownMenuTrigger asChild>
											<Button
												size="sm"
												className="bg-primary font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90"
											>
												Get Started
												<ChevronDown className="ml-1 h-4 w-4" />
											</Button>
										</DropdownMenuTrigger>
										<DropdownMenuContent align="end">
											<DropdownMenuItem>
												<SignUpButton signInForceRedirectUrl="/chat-platform">
													Chatbot App
												</SignUpButton>
											</DropdownMenuItem>
											<DropdownMenuItem>
												<SignUpButton signInForceRedirectUrl="/api-platform">
													API Platform
												</SignUpButton>
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>
								</SignedOut>

								<SignedIn>
									<DropdownMenu>
										<DropdownMenuTrigger asChild>
											<Button
												variant="ghost"
												size="sm"
												className="flex items-center gap-1 font-medium hover:bg-primary/50 hover:text-primary-foreground"
											>
												My Account
												<ChevronDown className="h-4 w-4" />
											</Button>
										</DropdownMenuTrigger>
										<DropdownMenuContent align="end">
											<DropdownMenuItem asChild>
												<Link href="/chat-platform">Chatbot App</Link>
											</DropdownMenuItem>
											<DropdownMenuItem asChild>
												<Link href="/api-platform">API Platform</Link>
											</DropdownMenuItem>
										</DropdownMenuContent>
									</DropdownMenu>
								</SignedIn>

								<ModeToggle />
							</div>
						</div>
					</div>
				</div>
			</nav>
		</header>
	);
}
