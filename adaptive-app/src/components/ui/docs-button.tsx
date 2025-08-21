import { HiOutlineDocumentText } from "react-icons/hi2";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface DocsButtonProps {
	variant?: "default" | "outline" | "ghost" | "link";
	size?: "default" | "sm" | "lg" | "icon";
	className?: string;
	showText?: boolean;
}

export function DocsButton({ 
	variant = "outline", 
	size = "sm", 
	className,
	showText = true 
}: DocsButtonProps) {
	const button = (
		<Button
			variant={variant}
			size={size}
			className={cn("", className)}
			asChild
		>
			<Link href="https://docs.llmadaptive.uk/" target="_blank" rel="noopener noreferrer">
				<HiOutlineDocumentText className="h-4 w-4" />
				{showText && <span className="ml-2">Docs</span>}
				<span className="sr-only">Documentation</span>
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
					<p>Documentation</p>
				</TooltipContent>
			</Tooltip>
		);
	}

	return button;
}

export function DocsLink({ className }: { className?: string }) {
	return (
		<Link 
			href="https://docs.llmadaptive.uk/" 
			target="_blank" 
			rel="noopener noreferrer"
			className={cn("text-primary hover:underline flex items-center gap-1", className)}
		>
			<HiOutlineDocumentText className="h-4 w-4" />
			Docs
		</Link>
	);
}