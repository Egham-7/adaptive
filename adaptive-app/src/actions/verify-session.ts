"use server";

import { stripe } from "@/lib/stripe/stripe";

export async function verifySession(sessionId: string) {
	try {
		const session = await stripe.checkout.sessions.retrieve(sessionId);

		return {
			isValid: session.payment_status === "paid",
			sessionData: {
				id: session.id,
				payment_status: session.payment_status,
				status: session.status,
				customer: session.customer,
				subscription: session.subscription,
				metadata: session.metadata,
			},
		};
	} catch (error) {
		return {
			isValid: false,
			error: "Failed to verify payment session",
		};
	}
}
