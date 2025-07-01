"use server";
import { stripe } from "@/lib/stripe/stripe";

if (!process.env.NEXT_PUBLIC_URL) {
	throw new Error("NEXT_PUBLIC_URL not found");
}

if (!process.env.STRIPE_CHAT_PRICE) {
	throw new Error("STRIPE_CHAT_PRICE not found");
}

type Props = {
	userId: string;
};

export const subscribeAction = async ({ userId }: Props) => {
	const _successUrl = `${process.env.NEXT_PUBLIC_URL}/chat-platform?success=true`;
	const _cancelUrl = `${process.env.NEXT_PUBLIC_URL}/?canceled=true`;

	const { url } = await stripe.checkout.sessions.create({
		payment_method_types: ["card"],
		line_items: [
			{
				price: process.env.STRIPE_CHAT_PRICE,
				quantity: 1,
			},
		],
		metadata: {
			userId,
		},
		mode: "subscription",
		success_url: `${process.env.NEXT_PUBLIC_URL}/chat-platform?success=true&session_id={CHECKOUT_SESSION_ID}`,
		cancel_url: `${process.env.NEXT_PUBLIC_URL}/?canceled=true`,
	});

	return url;
};
