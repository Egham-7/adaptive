import { stripe } from "@/lib/stripe/stripe";
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { db } from "@/server/db";

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

          console.log("✅ Subscription created/updated:", {
            userId: session.metadata.userId,
            subscriptionId: subscription.id,
            status: subscription.status
          });
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
        }
        break;

      case "customer.subscription.deleted":
        const deletedSubscription = event.data.object as Stripe.Subscription;
        
        await db.subscription.update({
          where: { stripeSubscriptionId: deletedSubscription.id },
          data: { 
            status: "canceled",
            currentPeriodEnd: new Date((deletedSubscription as any).current_period_end * 1000),
          },
        });
        break;

      case "invoice.payment_succeeded":
        const invoice = event.data.object as Stripe.Invoice;
        
        if ((invoice as any).subscription) {
          await db.subscription.update({
            where: { stripeSubscriptionId: (invoice as any).subscription as string },
            data: {
              status: "active",
              currentPeriodEnd: new Date(invoice.period_end * 1000),
            },
          });
        }
        break;

      case "invoice.payment_failed":
        const failedInvoice = event.data.object as Stripe.Invoice;
        
        if ((failedInvoice as any).subscription) {
          await db.subscription.update({
            where: { stripeSubscriptionId: (failedInvoice as any).subscription as string },
            data: { status: "past_due" },
          });
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
