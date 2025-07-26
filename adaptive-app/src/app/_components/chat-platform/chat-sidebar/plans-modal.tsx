"use client";

import { useState } from "react";
import {
	Dialog,
	DialogContent,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import ChatbotPricing from "../../landing_page/chatbot-pricing";

interface PlansModalProps {
	children: React.ReactNode;
}

export function PlansModal({ children }: PlansModalProps) {
	const [open, setOpen] = useState(false);

	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger asChild>{children}</DialogTrigger>
			<DialogContent className="!max-w-5xl w-full">
				<DialogTitle className="sr-only">Choose Your Plan</DialogTitle>
				<ChatbotPricing />
			</DialogContent>
		</Dialog>
	);
}
