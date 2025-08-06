"use client";

import { SignedIn, SignedOut, SignInButton, SignUpButton } from "@clerk/nextjs";
import { ChevronDown, Loader2 } from "lucide-react";
import Link, { useLinkStatus } from "next/link";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ModeToggle } from "../../mode-toggle";

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

export function UserActions() {
	return (
		<>
			{/* Separator */}
			<div className="h-4 w-px bg-border" />

			{/* User actions */}
			<div className="flex items-center gap-2">
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
									<Button variant="ghost" className="w-full justify-start">
										Chatbot App
									</Button>
								</LoadingLink>
							</DropdownMenuItem>
							<DropdownMenuItem asChild>
								<LoadingLink href="/api-platform/organizations">
									<Button variant="ghost" className="w-full justify-start">
										API Platform
									</Button>
								</LoadingLink>
							</DropdownMenuItem>
						</DropdownMenuContent>
					</DropdownMenu>
				</SignedIn>

				<ModeToggle />
			</div>
		</>
	);
}
