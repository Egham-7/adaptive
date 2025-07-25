"use client";

import { SignedIn, SignedOut, SignInButton, SignUpButton } from "@clerk/nextjs";
import { ChevronDown, Loader2, Menu, X } from "lucide-react";
import Link, { useLinkStatus } from "next/link";
import { useState } from "react";
import { FaGithub } from "react-icons/fa";
import { HiOutlineDocumentText } from "react-icons/hi2";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Logo } from "../logo";
import { ModeToggle } from "../mode-toggle";

const menuItems: MenuItem[] = [
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
		href: "https://docs.llmadaptive.uk/",
		icon: HiOutlineDocumentText,
		external: true,
	},
	{
		name: "GitHub",
		href: "https://github.com/Egham-7/adaptive",
		icon: FaGithub,
		external: true,
	},
];

function LoadingLink({
	href,
	children,
}: {
	href: string;
	children: React.ReactNode;
}) {
	const { pending } = useLinkStatus();

	return (
		<Link href={href} className="flex items-center gap-2">
			{pending && <Loader2 className="h-4 w-4 animate-spin" />}
			{children}
		</Link>
	);
}

export default function Header() {
	const [menuState, setMenuState] = useState(false);

	return (
		<header>
			<nav
				data-state={menuState && "active"}
				className="fixed z-20 w-full border-b border-dashed bg-background backdrop-blur-sm md:relative dark:bg-background/50 lg:dark:bg-transparent"
				aria-label="Main navigation"
			>
				<div className="m-auto max-w-5xl px-6">
					<div className="flex flex-wrap items-center justify-between gap-6 py-3 lg:gap-0 lg:py-4">
						<Link
							href="/"
							aria-label="Adaptive AI Home"
							className="flex items-center space-x-2"
						>
							<Logo />
						</Link>
						<button
							type="button"
							onClick={() => setMenuState(!menuState)}
							aria-label={menuState === true ? "Close Menu" : "Open Menu"}
							aria-expanded={menuState}
							className="-m-2.5 -mr-4 relative z-20 block cursor-pointer p-2.5 lg:hidden"
						>
							<Menu className="m-auto size-6 in-data-[state=active]:rotate-180 in-data-[state=active]:scale-0 in-data-[state=active]:opacity-0 duration-200" />
							<X className="-rotate-180 absolute inset-0 m-auto size-6 in-data-[state=active]:rotate-0 in-data-[state=active]:scale-100 scale-0 in-data-[state=active]:opacity-100 opacity-0 duration-200" />
						</button>

						<div className="hidden gap-8 text-sm lg:flex" role="menubar">
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
										<a
											href={item.href}
											className="cursor-pointer text-muted-foreground duration-150 hover:text-accent-foreground"
											role="menuitem"
											onClick={(e) => {
												e.preventDefault();
												const target = document.querySelector(item.href);
												target?.scrollIntoView({ behavior: "smooth" });
											}}
										>
											{item.name}
										</a>
									)}
								</div>
							))}
						</div>

						<nav
							className="hidden lg:flex lg:items-center lg:gap-3"
							aria-label="External links"
						>
							{iconMenuItems.map((item) => {
								const Icon = item.icon;
								return (
									<a
										key={item.name}
										href={item.href}
										target="_blank"
										rel="noopener noreferrer"
										className="flex items-center gap-2 text-muted-foreground duration-150 hover:text-accent-foreground"
										aria-label={`Visit ${item.name}`}
									>
										<Icon size={20} aria-hidden="true" />
										<span className="text-sm">{item.name}</span>
									</a>
								);
							})}
						</nav>

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
										<DropdownMenuItem asChild>
											<SignInButton signUpForceRedirectUrl="/chat-platform">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													Chatbot App
												</Button>
											</SignInButton>
										</DropdownMenuItem>
										<DropdownMenuItem asChild>
											<SignInButton signUpForceRedirectUrl="/api-platform/organizations">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													API Platform
												</Button>
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
										<DropdownMenuItem asChild>
											<SignUpButton signInForceRedirectUrl="/chat-platform">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													Chatbot App
												</Button>
											</SignUpButton>
										</DropdownMenuItem>
										<DropdownMenuItem asChild>
											<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													API Platform
												</Button>
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
											<LoadingLink href="/chat-platform">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													Chatbot App
												</Button>
											</LoadingLink>
										</DropdownMenuItem>
										<DropdownMenuItem asChild>
											<LoadingLink href="/api-platform/organizations">
												<Button
													variant="ghost"
													className="w-full justify-start"
												>
													API Platform
												</Button>
											</LoadingLink>
										</DropdownMenuItem>
									</DropdownMenuContent>
								</DropdownMenu>
							</SignedIn>

							<ModeToggle />
						</div>

						{/* Mobile menu - keep existing mobile menu structure */}
						<div className="mb-6 in-data-[state=active]:block hidden w-full flex-col items-center space-y-6 rounded-3xl border bg-background p-6 shadow-2xl shadow-color-shadow-color/[var(--shadow-opacity)] lg:hidden">
							<div className="mb-8 w-full">
								<ul className="space-y-4 text-center text-base">
									{menuItems.map((item) => (
										<li key={item.name}>
											{item.external ? (
												<a
													href={item.href}
													target="_blank"
													rel="noopener noreferrer"
													className="block text-center text-muted-foreground duration-150 hover:text-accent-foreground"
												>
													<span>{item.name}</span>
												</a>
											) : (
												<a
													href={item.href}
													className="block cursor-pointer text-center text-muted-foreground duration-150 hover:text-accent-foreground"
													onClick={(e) => {
														e.preventDefault();
														const target = document.querySelector(item.href);
														target?.scrollIntoView({ behavior: "smooth" });
														setMenuState(false);
													}}
												>
													<span>{item.name}</span>
												</a>
											)}
										</li>
									))}
								</ul>

								{/* Icon links for mobile */}
								<div className="mt-8 mb-8 flex justify-center gap-6 border-t border-dashed pt-8">
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
												<Icon size={20} />
												<span>{item.name}</span>
											</a>
										);
									})}
								</div>
							</div>

							<div className="mt-8 flex w-full flex-col items-center space-y-3 border-t border-dashed pt-8">
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
											<DropdownMenuItem asChild>
												<SignInButton signUpForceRedirectUrl="/chat-platform">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														Chatbot App
													</Button>
												</SignInButton>
											</DropdownMenuItem>
											<DropdownMenuItem asChild>
												<SignInButton signUpForceRedirectUrl="/api-platform/organizations">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														API Platform
													</Button>
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
											<DropdownMenuItem asChild>
												<SignUpButton signInForceRedirectUrl="/chat-platform">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														Chatbot App
													</Button>
												</SignUpButton>
											</DropdownMenuItem>
											<DropdownMenuItem asChild>
												<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														API Platform
													</Button>
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
												<LoadingLink href="/chat-platform">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														Chatbot App
													</Button>
												</LoadingLink>
											</DropdownMenuItem>
											<DropdownMenuItem asChild>
												<LoadingLink href="/api-platform/organizations">
													<Button
														variant="ghost"
														className="w-full justify-start"
													>
														API Platform
													</Button>
												</LoadingLink>
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
