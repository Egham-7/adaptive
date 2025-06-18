"use client";

import { subscribeAction } from "@/actions/subscribe";
import { cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { useTransition } from "react";

type Props = {
	userId: string;
	children?: React.ReactNode;
	className?: string;
	variant?: "default" | "link";
	disabled?: boolean;
};

function SubscribeButton({
	userId,
	children,
	className,
	variant = "default",
	disabled = false,
}: Props) {
	const router = useRouter();
	const [isPending, startTransition] = useTransition();

	const handleClickSubscribeButton = async () => {
		startTransition(async () => {
			const url = await subscribeAction({ userId });
			if (url) {
				router.push(url);
			} else {
				console.error("Failed to create subscription session");
			}
		});
	};

	const isDisabled = isPending || disabled;

	if (variant === "link") {
		return (
			<button
				type="button"
				disabled={isDisabled}
				onClick={handleClickSubscribeButton}
				className={cn(
					"font-semibold underline hover:opacity-80 disabled:opacity-50",
					className,
				)}
			>
				{isPending ? "Loading..." : children || "Upgrade to Pro"}
			</button>
		);
	}

	return (
		<button
			type="button"
			disabled={isDisabled}
			onClick={handleClickSubscribeButton}
			className={cn(
				"w-full rounded-lg bg-primary p-2 text-center font-medium text-primary-foreground shadow-subtle transition-opacity hover:opacity-90 disabled:opacity-50",
				className,
			)}
		>
			{isPending ? "Loading..." : children || "Subscribe"}
		</button>
	);
}

export default SubscribeButton;
