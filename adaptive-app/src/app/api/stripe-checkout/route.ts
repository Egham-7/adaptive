import { stripe } from "@/lib/stripe/stripe";
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { db } from "@/server/db";
import { invalidateSubscriptionCache } from "@/lib/cache-utils";

export async function POST(request: NextRequest) {
  const body = await request.text();
  const headersList = await headers();
  const signature = headersList.get("Stripe-Signature") as string;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_SIGNING_SECRET!
    );
  } catch (error) {
    console.error("❌ Webhook signature verification failed:", error);
    return new NextResponse("webhook error", { status: 400 });
  }

  try {
    switch (event.type) {
      case "checkout.session.completed":
        if (event.data.object.payment_status === "paid") {
          const session = event.data.object as Stripe.Checkout.Session;
          
          // Skip if missing required fields (test data)
          if (!session.metadata?.userId || !session.subscription || !session.customer) {
            break;
          }

          // Get the subscription details from Stripe
          const subscription = await stripe.subscriptions.retrieve(
            session.subscription as string
          );

          // Calculate next month's date for the subscription period
          const currentPeriodEnd = new Date();
          currentPeriodEnd.setMonth(currentPeriodEnd.getMonth() + 1);

          const priceId = subscription.items.data[0]?.price.id;

          if (!priceId) {
            throw new Error("Subscription price ID is missing");
          }

          // Ensure customer ID is a string
          const customerId = typeof session.customer === 'string' 
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

          // Invalidate subscription cache for this user
          await invalidateSubscriptionCache(session.metadata.userId);

          console.log("✅ Subscription created/updated successfully");
        }
        break;

      case "customer.subscription.updated":
        const subscription = event.data.object as Stripe.Subscription;
        
        const existingSub = await db.subscription.findUnique({
          where: { stripeSubscriptionId: subscription.id },
        });

        if (existingSub) {
          await db.subscription.update({
            where: { stripeSubscriptionId: subscription.id },
            data: {
              status: subscription.status,
              currentPeriodEnd: new Date((subscription as any).current_period_end * 1000),
              stripePriceId: subscription.items.data[0]?.price.id || existingSub.stripePriceId,
            },
          });

          // Invalidate subscription cache for this user
          await invalidateSubscriptionCache(existingSub.userId);

          console.log("✅ Subscription updated successfully");
        }
        break;

      case "customer.subscription.deleted":
        const deletedSubscription = event.data.object as Stripe.Subscription;
        
        const canceledSub = await db.subscription.update({
          where: { stripeSubscriptionId: deletedSubscription.id },
          data: { 
            status: "canceled",
            currentPeriodEnd: new Date((deletedSubscription as any).current_period_end * 1000),
          },
        });

        // Invalidate subscription cache for this user
        await invalidateSubscriptionCache(canceledSub.userId);

        console.log("✅ Subscription canceled successfully");
        break;

      case "invoice.payment_succeeded":
        const invoice = event.data.object as Stripe.Invoice;
        
        if ((invoice as any).subscription) {
          const subscriptionId = (invoice as any).subscription as string;
          
          // Check if subscription exists before updating
          const existingSubscription = await db.subscription.findUnique({
            where: { stripeSubscriptionId: subscriptionId },
          });

          if (existingSubscription) {
            await db.subscription.update({
              where: { stripeSubscriptionId: subscriptionId },
              data: {
                status: "active",
                currentPeriodEnd: new Date(invoice.period_end * 1000),
              },
            });

            // Invalidate subscription cache for this user
            await invalidateSubscriptionCache(existingSubscription.userId);

            console.log("✅ Subscription activated successfully");
          }
        }
        break;

      case "invoice.payment_failed":
        const failedInvoice = event.data.object as Stripe.Invoice;
        
        if ((failedInvoice as any).subscription) {
          const subscriptionId = (failedInvoice as any).subscription as string;
          
          // Check if subscription exists before updating
          const existingFailedSubscription = await db.subscription.findUnique({
            where: { stripeSubscriptionId: subscriptionId },
          });

          if (existingFailedSubscription) {
            await db.subscription.update({
              where: { stripeSubscriptionId: subscriptionId },
              data: { status: "past_due" },
            });

            // Invalidate subscription cache for this user
            await invalidateSubscriptionCache(existingFailedSubscription.userId);

            console.log("⚠️ Subscription marked past due");
          }
        }
        break;
    }
  } catch (error) {
    console.error("❌ Webhook error:", error);
    return new NextResponse("Error processing webhook", { status: 500 });
  }

  revalidatePath("/", "layout");
  return new NextResponse(null, { status: 200 });
}