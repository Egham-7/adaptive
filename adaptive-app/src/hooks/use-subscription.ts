"use client";

import { useRouter } from "next/navigation";
import { api } from "@/trpc/react";

export function useSubscription() {
	const router = useRouter();
	const utils = api.useUtils();

	const cancelSubscription = api.subscription.cancelSubscription.useMutation({
		onSuccess: () => {
			// Invalidate all subscription-related queries globally
			utils.subscription.invalidate();
			router.push("/chat-platform");
		},
		onError: (error) => {
			console.error("Failed to cancel subscription:", error);
		},
	});

	const createCheckoutSession =
		api.subscription.createCheckoutSession.useMutation({
			onSuccess: (data) => {
				// Redirect to Stripe checkout
				if (typeof window !== "undefined") {
					window.location.href = data.url;
				}
			},
			onError: (error) => {
				console.error("Failed to create checkout session:", error);
			},
		});

	return {
		cancelSubscription,
		createCheckoutSession,
	};
}
