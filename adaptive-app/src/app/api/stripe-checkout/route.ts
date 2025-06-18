import { stripe } from "@/lib/stripe/stripe";
import { db } from "@/server/db";
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";
import type Stripe from "stripe";

export async function POST(request: NextRequest) {
	const body = await request.text();
	const headersList = await headers();
	const signature = headersList.get("stripe-signature");

	let event: Stripe.Event;

	// Check if webhook signing is configured
	if (!process.env.STRIPE_SIGNING_SECRET) {
		console.error("❌ Webhook secret not configured");
		return new NextResponse("Webhook secret not configured", { status: 500 });
	}

	if (!signature) {
		console.error("❌ No stripe signature found");
		return new NextResponse("No signature found", { status: 400 });
	}

	try {
		// Verify the webhook signature using the raw body and secret
		event = stripe.webhooks.constructEvent(
			body,
			signature,
			process.env.STRIPE_SIGNING_SECRET,
		);
	} catch (err) {
		console.error("❌ Webhook signature verification failed:", err);
		return new NextResponse("Webhook signature verification failed", {
			status: 400,
		});
	}

	// Extract the object from the event
	const data = event.data;
	const eventType = event.type;

	try {
		switch (eventType) {
			case "checkout.session.completed": {
				const session = data.object as Stripe.Checkout.Session;
				console.log("🔔 Payment received!");

				if (session.payment_status === "paid") {
					// Skip if missing required fields
					if (
						!session.metadata?.userId ||
						!session.subscription ||
						!session.customer
					) {
						console.log("⚠️ Missing required session data, skipping...");
						break;
					}

					// Get the subscription details from Stripe API
					const subscription = await stripe.subscriptions.retrieve(
						session.subscription as string,
					);
					const currentPeriodEnd = new Date(
						// @ts-ignore - current_period_end exists in Stripe API but not in TS types
						subscription.current_period_end * 1000,
					);
					const priceId = subscription.items.data[0]?.price.id;

					if (!priceId) {
						throw new Error("Subscription price ID is missing");
					}

					// Ensure customer ID is a string
					const customerId =
						typeof session.customer === "string"
							? session.customer
							: session.customer.id;

					// Update or create subscription in database
					await db.subscription.upsert({
						where: {
							userId: session.metadata.userId,
						},
						create: {
							userId: session.metadata.userId,
							stripeCustomerId: customerId,
							stripePriceId: priceId,
							stripeSubscriptionId: subscription.id,
							status: subscription.status,
							currentPeriodEnd,
						},
						update: {
							stripeCustomerId: customerId,
							stripePriceId: priceId,
							stripeSubscriptionId: subscription.id,
							status: subscription.status,
							currentPeriodEnd,
						},
					});

					console.log("✅ Subscription created/updated:", {
						userId: session.metadata.userId,
						subscriptionId: subscription.id,
						status: subscription.status,
					});
				}
				break;
			}

			case "customer.subscription.updated": {
				const subscription = data.object as Stripe.Subscription;
				console.log("🔄 Subscription updated");

				const existingSub = await db.subscription.findUnique({
					where: { stripeSubscriptionId: subscription.id },
				});

				if (existingSub) {
					// Get complete subscription data to ensure we have all properties
					const fullSubscription = await stripe.subscriptions.retrieve(
						subscription.id,
					);

					await db.subscription.update({
						where: { stripeSubscriptionId: subscription.id },
						data: {
							status: subscription.status,
							// @ts-ignore - current_period_end exists in Stripe API but not in TS types
							currentPeriodEnd: new Date(
								fullSubscription.current_period_end * 1000,
							),
							stripePriceId:
								subscription.items.data[0]?.price.id ||
								existingSub.stripePriceId,
						},
					});

					console.log("✅ Subscription updated:", subscription.id);
				}
				break;
			}

			case "customer.subscription.deleted": {
				const subscription = data.object as Stripe.Subscription;
				console.log("🗑️ Subscription deleted");

				await db.subscription.update({
					where: { stripeSubscriptionId: subscription.id },
					data: {
						status: "canceled",
					},
				});

				console.log("✅ Subscription canceled:", subscription.id);
				break;
			}

			case "invoice.payment_succeeded": {
				const invoice = data.object as Stripe.Invoice;
				console.log("💰 Invoice payment succeeded");

				// Handle successful invoice payment
				if (invoice.lines?.data?.length > 0) {
					for (const line of invoice.lines.data) {
						if (line.subscription) {
							const subscriptionId =
								typeof line.subscription === "string"
									? line.subscription
									: line.subscription.id;

							await db.subscription.update({
								where: { stripeSubscriptionId: subscriptionId },
								data: {
									status: "active",
									currentPeriodEnd: new Date(invoice.period_end * 1000),
								},
							});

							console.log("✅ Subscription activated:", subscriptionId);
							break;
						}
					}
				}
				break;
			}

			case "invoice.payment_failed": {
				const invoice = data.object as Stripe.Invoice;
				console.log("❌ Invoice payment failed");

				// Handle failed invoice payment
				if (invoice.lines?.data?.length > 0) {
					for (const line of invoice.lines.data) {
						if (line.subscription) {
							const subscriptionId =
								typeof line.subscription === "string"
									? line.subscription
									: line.subscription.id;

							await db.subscription.update({
								where: { stripeSubscriptionId: subscriptionId },
								data: {
									status: "past_due",
								},
							});

							console.log("⚠️ Subscription marked past due:", subscriptionId);
							break;
						}
					}
				}
				break;
			}

			default:
				console.log(`🤷‍♂️ Unhandled event type: ${eventType}`);
		}
	} catch (error) {
		console.error("❌ Webhook handler error:", error);
		return new NextResponse("Webhook handler error", { status: 500 });
	}

	revalidatePath("/", "layout");
	return new NextResponse(null, { status: 200 });
}
