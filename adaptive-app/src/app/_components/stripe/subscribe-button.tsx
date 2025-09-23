"use client";

import { useRouter } from "next/navigation";
import { cn } from "@/lib/shared/utils";
import { api } from "@/trpc/react";

type Props = {
	children?: React.ReactNode;
	className?: string;
	variant?: "default" | "link";
	disabled?: boolean;
};

function SubscribeButton({
	children,
	className,
	variant = "default",
	disabled = false,
}: Props) {
	const router = useRouter();
	const createCheckoutSession =
		api.subscription.createCheckoutSession.useMutation();

	const handleClickSubscribeButton = async () => {
		try {
			const result = await createCheckoutSession.mutateAsync({});
			if (result.url) {
				router.push(result.url);
			} else {
				console.error("Failed to create subscription session");
			}
		} catch (error) {
			console.error("Failed to create subscription session:", error);
		}
	};

	const isDisabled = createCheckoutSession.isPending || disabled;

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
				{createCheckoutSession.isPending
					? "Loading..."
					: children || "Upgrade to Pro"}
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
			{createCheckoutSession.isPending ? "Loading..." : children || "Subscribe"}
		</button>
	);
}

export default SubscribeButton;
