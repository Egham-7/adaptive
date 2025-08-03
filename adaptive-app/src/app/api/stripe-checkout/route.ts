import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";
import type Stripe from "stripe";
import { addCredits } from "@/lib/credit-utils";
import { stripe } from "@/lib/stripe/stripe";
import { db } from "@/server/db";

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
		console.log("🔄 Processing webhook event:");

		switch (eventType) {
			case "checkout.session.completed": {
				const session = data.object as Stripe.Checkout.Session;
				console.log("🔔 Payment received!");

				if (session.payment_status === "paid") {
					// Check if this is a credit purchase or subscription
					const isSubscription = !!session.subscription;
					const isCreditPurchase = session.metadata?.type === "credit_purchase";

					if (!session.metadata?.userId || !session.customer) {
						console.log("⚠️ Missing required session data, skipping...");
						break;
					}

					if (isCreditPurchase) {
						// Handle credit purchase
						console.log("💳 Processing credit purchase");

						if (!session.metadata?.organizationId) {
							console.error("❌ Missing organizationId for credit purchase");
							break;
						}

						const creditAmount = Number.parseFloat(
							session.metadata.creditAmount || "0",
						);
						if (creditAmount <= 0) {
							console.error(
								"❌ Invalid credit amount:",
								session.metadata.creditAmount,
							);
							break;
						}

						try {
							console.log("💳 Adding credits to organization.");

							// Add credits to organization's account
							const _result = await addCredits({
								organizationId: session.metadata.organizationId,
								userId: session.metadata.userId,
								amount: creditAmount,
								type: "purchase",
								description: `Credit purchase via Stripe - $${creditAmount}`,
								metadata: {
									stripeSessionId: session.id,
									stripePaymentIntentId: session.payment_intent,
									purchaseTimestamp: new Date().toISOString(),
								},
								stripeSessionId: session.id,
								stripePaymentIntentId: session.payment_intent as string,
							});

							console.log("✅ Credits added successfully.");
						} catch (_error) {
							console.error("❌ Failed to add credits.");
						}
					} else if (isSubscription) {
						// Handle subscription (existing logic)
						if (!session.subscription) {
							console.log("⚠️ Missing subscription data, skipping...");
							break;
						}

						// Get the subscription details from Stripe API
						const subscription = await stripe.subscriptions.retrieve(
							session.subscription as string,
						);

						const subscriptionItem = subscription.items.data[0];
						if (!subscriptionItem) {
							throw new Error("No subscription items found");
						}

						const currentPeriodEnd = new Date(
							subscriptionItem.current_period_end * 1000,
						);
						const priceId = subscriptionItem.price.id;

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

					const subscriptionItem = fullSubscription.items.data[0];
					if (!subscriptionItem) {
						throw new Error(
							"No subscription items found in updated subscription",
						);
					}

					await db.subscription.update({
						where: { stripeSubscriptionId: subscription.id },
						data: {
							status: subscription.status,
							currentPeriodEnd: new Date(
								subscriptionItem.current_period_end * 1000,
							),
							stripePriceId: subscriptionItem.price.id,
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

							await db.subscription.updateMany({
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

							await db.subscription.updateMany({
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
