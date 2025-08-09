import { HelpCircle } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
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
	return (
		<Button
			variant={variant}
			size={size}
			className={cn("", className)}
			asChild
		>
			<Link href="/support" target="_blank" rel="noopener noreferrer">
				<HelpCircle className="h-4 w-4" />
				{showText && <span className="ml-2">Support</span>}
			</Link>
		</Button>
	);
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