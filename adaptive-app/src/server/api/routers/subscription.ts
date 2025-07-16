import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { invalidateSubscriptionCache, withCache } from "@/lib/cache-utils";
import { stripe } from "@/lib/stripe/stripe";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

export const subscriptionRouter = createTRPCRouter({
	// Get current user's subscription status
	getSubscription: protectedProcedure.query(async ({ ctx }) => {
		const userId = ctx.clerkAuth.userId;
		const cacheKey = `subscription:${userId}`;

		return withCache(cacheKey, async () => {
			const subscription = await ctx.db.subscription.findFirst({
				where: {
					userId: userId,
				},
			});

			if (!subscription) {
				return { subscribed: false, subscription: null };
			}

			return {
				subscribed: subscription.status === "active",
				subscription: {
					id: subscription.id,
					status: subscription.status,
					currentPeriodEnd: subscription.currentPeriodEnd,
					stripePriceId: subscription.stripePriceId,
				},
			};
		});
	}),

	// Create Stripe checkout session
	createCheckoutSession: protectedProcedure
		.input(
			z.object({
				priceId: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				const { priceId = process.env.STRIPE_CHAT_PRICE } = input;

				if (!priceId) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Price ID is required",
					});
				}

				if (!process.env.NEXT_PUBLIC_URL) {
					throw new TRPCError({
						code: "INTERNAL_SERVER_ERROR",
						message: "Base URL not configured",
					});
				}

				const session = await stripe.checkout.sessions.create({
					payment_method_types: ["card"],
					line_items: [
						{
							price: priceId,
							quantity: 1,
						},
					],
					metadata: {
						userId: ctx.clerkAuth.userId,
					},
					mode: "subscription",
					success_url: `${process.env.NEXT_PUBLIC_URL}/chat-platform?success=true&session_id={CHECKOUT_SESSION_ID}`,
					cancel_url: `${process.env.NEXT_PUBLIC_URL}/?canceled=true`,
				});

				if (!session.url) {
					throw new TRPCError({
						code: "INTERNAL_SERVER_ERROR",
						message: "Failed to create checkout session",
					});
				}

				return {
					url: session.url,
					sessionId: session.id,
				};
			} catch (error) {
				console.error("Checkout session creation error:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message:
						process.env.NODE_ENV === "development"
							? `Failed to create checkout session: ${error instanceof Error ? error.message : String(error)}`
							: "Failed to create checkout session",
				});
			}
		}),

	// Verify payment session
	verifySession: protectedProcedure
		.input(
			z.object({
				sessionId: z.string(),
			}),
		)
		.mutation(async ({ input }) => {
			try {
				const session = await stripe.checkout.sessions.retrieve(
					input.sessionId,
				);

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
				console.error("Session verification error:", error);
				throw new TRPCError({
					code: "BAD_REQUEST",
					message: "Failed to verify payment session",
				});
			}
		}),

	// Cancel subscription
	cancelSubscription: protectedProcedure.mutation(async ({ ctx }) => {
		const subscription = await ctx.db.subscription.findFirst({
			where: {
				userId: ctx.clerkAuth.userId,
				status: "active",
			},
		});

		if (!subscription) {
			throw new TRPCError({
				code: "NOT_FOUND",
				message: "No active subscription found",
			});
		}

		if (!subscription.stripeSubscriptionId) {
			throw new TRPCError({
				code: "BAD_REQUEST",
				message: "Invalid subscription data",
			});
		}

		try {
			// Cancel the subscription in Stripe
			await stripe.subscriptions.cancel(subscription.stripeSubscriptionId);

			// Update the subscription status in the database
			await ctx.db.subscription.update({
				where: {
					id: subscription.id,
				},
				data: {
					status: "canceled",
				},
			});

			// Invalidate subscription cache
			await invalidateSubscriptionCache(ctx.clerkAuth.userId);

			return {
				success: true,
				message: "Subscription canceled successfully",
			};
		} catch (error) {
			console.error("Subscription cancellation error:", error);
			throw new TRPCError({
				code: "INTERNAL_SERVER_ERROR",
				message: "Failed to cancel subscription",
			});
		}
	}),

	// Check if user is subscribed (utility function)
	isSubscribed: protectedProcedure.query(async ({ ctx }) => {
		const userId = ctx.clerkAuth.userId;
		const cacheKey = `subscription-status:${userId}`;

		return withCache(cacheKey, async () => {
			const subscription = await ctx.db.subscription.findFirst({
				where: {
					userId: userId,
					status: "active",
				},
			});

			return !!subscription;
		});
	}),
});
