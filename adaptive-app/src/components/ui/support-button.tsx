import { HelpCircle } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface SupportButtonProps {
	variant?: "default" | "outline" | "ghost" | "link";
	size?: "default" | "sm" | "lg" | "icon";
	className?: string;
	showText?: boolean;
}

export function SupportButton({ 
	variant = "outline", 
	size = "sm", 
	className,
	showText = true 
}: SupportButtonProps) {
	const button = (
		<Button
			variant={variant}
			size={size}
			className={cn("", className)}
			asChild
		>
			<Link href="/support" target="_blank" rel="noopener noreferrer">
				<HelpCircle className="h-4 w-4" />
				{showText && <span className="ml-2">Support</span>}
				<span className="sr-only">Support</span>
			</Link>
		</Button>
	);

	// Add tooltip when text is hidden (icon-only)
	if (!showText) {
		return (
			<Tooltip>
				<TooltipTrigger asChild>
					{button}
				</TooltipTrigger>
				<TooltipContent>
					<p>Support</p>
				</TooltipContent>
			</Tooltip>
		);
	}

	return button;
}

export function SupportLink({ className }: { className?: string }) {
	return (
		<Link 
			href="/support" 
			target="_blank" 
			rel="noopener noreferrer"
			className={cn("text-primary hover:underline flex items-center gap-1", className)}
		>
			<HelpCircle className="h-4 w-4" />
			Support
		</Link>
	);
}