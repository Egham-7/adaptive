"use client";

import { SignedIn, SignedOut, SignInButton, SignUpButton } from "@clerk/nextjs";
import { ChevronDown } from "lucide-react";
import Link, { useLinkStatus } from "next/link";
import { usePathname } from "next/navigation";
import { GitHubStarsButton } from "@/components/animate-ui/buttons/github-stars";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useSmartRedirect } from "@/hooks/use-smart-redirect";
import { ModeToggle } from "../../mode-toggle";
import { iconMenuItems, menuItems } from "./navigation-items";

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
			{pending && <div className="h-4 w-4 animate-spin" />}
			{children}
		</Link>
	);
}

interface MobileNavigationProps {
	menuState: boolean;
	setMenuState: (state: boolean) => void;
}

export function MobileNavigation({
	menuState,
	setMenuState,
}: MobileNavigationProps) {
	const pathname = usePathname();
	const redirectPath = useSmartRedirect();

	return (
		<div
			className={`fixed inset-0 z-25 ${menuState ? "block" : "hidden"} bg-background/70 backdrop-blur-md lg:hidden`}
		>
			<button
				type="button"
				className="absolute inset-0 cursor-default"
				onClick={() => setMenuState(false)}
				aria-label="Close menu"
			/>
			<div className="relative flex min-h-full flex-col items-center justify-center space-y-8 p-6">
				<ul className="space-y-4 text-center text-base">
					{menuItems.map((item) => {
						const isActive =
							item.href === "/"
								? pathname === item.href
								: pathname.startsWith(item.href);
						return (
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
									<Link
										href={item.href}
										className={`block text-center duration-150 hover:text-accent-foreground ${
											isActive
												? "font-medium text-primary"
												: "text-muted-foreground"
										}`}
										onClick={() => setMenuState(false)}
										aria-current={isActive ? "page" : undefined}
									>
										<span>{item.name}</span>
									</Link>
								)}
							</li>
						);
					})}
				</ul>

				{/* Icon links and GitHub stars for mobile */}
				<div className="flex flex-col items-center gap-4 border-t border-dashed pt-6">
					<div className="flex justify-center gap-6">
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
					<GitHubStarsButton
						username="Egham-7"
						repo="adaptive"
						formatted={true}
						className="text-xs"
					/>
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
									<Button variant="ghost" className="w-full justify-start">
										Chatbot App
									</Button>
								</SignInButton>
							</DropdownMenuItem>
							<DropdownMenuItem asChild>
								<SignInButton signUpForceRedirectUrl="/api-platform/organizations">
									<Button variant="ghost" className="w-full justify-start">
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
									<Button variant="ghost" className="w-full justify-start">
										Chatbot App
									</Button>
								</SignUpButton>
							</DropdownMenuItem>
							<DropdownMenuItem asChild>
								<SignUpButton signInForceRedirectUrl="/api-platform/organizations">
									<Button variant="ghost" className="w-full justify-start">
										API Platform
									</Button>
								</SignUpButton>
							</DropdownMenuItem>
						</DropdownMenuContent>
					</DropdownMenu>
				</SignedOut>

				<SignedIn>
					<div className="flex w-full flex-col items-center gap-3">
						<LoadingLink href="/chat-platform">
							<Button
								variant="outline"
								className="w-full"
								onClick={() => setMenuState(false)}
							>
								Chatbot App
							</Button>
						</LoadingLink>
						<LoadingLink href={redirectPath || "/api-platform/organizations"}>
							<Button
								variant="outline"
								className="w-full"
								onClick={() => setMenuState(false)}
							>
								API Platform
							</Button>
						</LoadingLink>
					</div>
				</SignedIn>

				<ModeToggle />
			</div>
		</div>
	);
}
