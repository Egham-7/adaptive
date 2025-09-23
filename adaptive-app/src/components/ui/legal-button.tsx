import { Scale } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/shared/utils";

interface LegalButtonProps {
	variant?: "default" | "outline" | "ghost" | "link";
	size?: "default" | "sm" | "lg" | "icon";
	className?: string;
	showText?: boolean;
}

export function LegalButton({ 
	variant = "outline", 
	size = "sm", 
	className,
	showText = true 
}: LegalButtonProps) {
	if (!showText) {
		// When icon-only, use dropdown menu with tooltip
		return (
			<Tooltip>
				<DropdownMenu>
					<TooltipTrigger asChild>
						<DropdownMenuTrigger asChild>
							<Button
								variant={variant}
								size={size}
								className={cn("", className)}
							>
								<Scale className="h-4 w-4" />
								<span className="sr-only">Legal</span>
							</Button>
						</DropdownMenuTrigger>
					</TooltipTrigger>
					<DropdownMenuContent align="end">
						<DropdownMenuItem asChild>
							<Link href="/terms-of-service" target="_blank" rel="noopener noreferrer">
								Terms of Service
							</Link>
						</DropdownMenuItem>
						<DropdownMenuItem asChild>
							<Link href="/privacy-policy" target="_blank" rel="noopener noreferrer">
								Privacy Policy
							</Link>
						</DropdownMenuItem>
					</DropdownMenuContent>
				</DropdownMenu>
				<TooltipContent>
					<p>Legal</p>
				</TooltipContent>
			</Tooltip>
		);
	}

	// When showing text, use dropdown menu
	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button
					variant={variant}
					size={size}
					className={cn("", className)}
				>
					<Scale className="h-4 w-4" />
					{showText && <span className="ml-2">Legal</span>}
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="end">
				<DropdownMenuItem asChild>
					<Link href="/terms-of-service" target="_blank" rel="noopener noreferrer">
						Terms of Service
					</Link>
				</DropdownMenuItem>
				<DropdownMenuItem asChild>
					<Link href="/privacy-policy" target="_blank" rel="noopener noreferrer">
						Privacy Policy
					</Link>
				</DropdownMenuItem>
			</DropdownMenuContent>
		</DropdownMenu>
	);
}