// app/api/stripe/webhook/route.ts
import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

import Stripe from "stripe"; // ✅ types and Stripe class
import { stripe } from "@/lib/stripe/stripe";
import { db } from "@/server/db";

export async function POST(request: NextRequest) {
  const body = await request.text();
  const headersList = await headers();
  const signature = headersList.get("stripe-signature");

  if (!signature || !process.env.STRIPE_SIGNING_SECRET) {
    return new NextResponse("Missing Stripe signature or secret", {
      status: 400,
    });
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_SIGNING_SECRET
    );
  } catch (err) {
    console.error("❌ Webhook verification failed", err);
    return new NextResponse("Invalid signature", { status: 400 });
  }

  const data = event.data;
  const eventType = event.type;

  const loggableEvents = new Set([
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "invoice.payment_succeeded",
  ]);

  try {
    switch (eventType) {
      case "checkout.session.completed": {
        const session = data.object as Stripe.Checkout.Session;

        if (
          session.payment_status === "paid" &&
          session.subscription &&
          session.customer &&
          session.metadata?.userId
        ) {
          const subscriptionId = session.subscription as string;

          const subscription: Stripe.Subscription =
            await stripe.subscriptions.retrieve(subscriptionId);

          const currentPeriodEnd = getSafeCurrentPeriodEnd(subscription);

          const priceId = subscription.items.data[0]?.price.id;
          if (!priceId) throw new Error("Missing subscription price ID");

          const customerId =
            typeof session.customer === "string"
              ? session.customer
              : session.customer.id;

          await db.subscription.upsert({
            where: { userId: session.metadata.userId },
            create: {
              userId: session.metadata.userId,
              stripeCustomerId: customerId,
              stripeSubscriptionId: subscription.id,
              stripePriceId: priceId,
              status: subscription.status,
              currentPeriodEnd,
            },
            update: {
              stripeCustomerId: customerId,
              stripeSubscriptionId: subscription.id,
              stripePriceId: priceId,
              status: subscription.status,
              currentPeriodEnd,
            },
          });

          console.log("✅ Subscription upserted:", subscription.id);
        }
        break;
      }

      case "customer.subscription.updated": {
        const sub = data.object as Stripe.Subscription;

        // 🔍 Make sure the subscription exists in the DB
        const existing = await db.subscription.findUnique({
          where: { stripeSubscriptionId: sub.id },
        });

        // ✅ Exit early if it doesn't exist (could have been deleted manually)
        if (!existing) {
          console.warn("⚠️ Tried to update non-existing subscription:", sub.id);
          break; // ⛔ don't continue
        }

        // 🔄 Get fresh subscription from Stripe to ensure full data
        const full = await stripe.subscriptions.retrieve(sub.id);

        const currentPeriodEnd = getSafeCurrentPeriodEnd(full);

        await db.subscription.update({
          where: { stripeSubscriptionId: sub.id },
          data: {
            status: full.status,
            currentPeriodEnd,
            stripePriceId:
              full.items.data[0]?.price.id || existing.stripePriceId,
          },
        });

        console.log("🔄 Subscription updated:", sub.id);
        break;
      }

      case "customer.subscription.deleted": {
        const sub = data.object as Stripe.Subscription;

        await db.subscription.update({
          where: { stripeSubscriptionId: sub.id },
          data: { status: "canceled" },
        });

        console.log("🗑️ Subscription canceled:", sub.id);
        break;
      }

      case "invoice.payment_succeeded": {
        const invoice = data.object as Stripe.Invoice;

        for (const line of invoice.lines.data) {
          const subId =
            typeof line.subscription === "string"
              ? line.subscription
              : line.subscription?.id;

          if (!subId) continue;

          const existing = await db.subscription.findUnique({
            where: { stripeSubscriptionId: subId },
          });

          if (!existing) {
            console.warn("⚠️ Skipping update — subscription not found:", subId);
            continue;
          }

          await db.subscription.update({
            where: { stripeSubscriptionId: subId },
            data: {
              status: "active",
              currentPeriodEnd: new Date(invoice.period_end * 1000),
            },
          });

          console.log("💰 Payment succeeded:", subId);
          break;
        }
        break;
      }

      case "invoice.payment_failed": {
        const invoice = data.object as Stripe.Invoice;

        for (const line of invoice.lines.data) {
          const subId =
            typeof line.subscription === "string"
              ? line.subscription
              : line.subscription?.id;

          if (!subId) continue;

          await db.subscription.update({
            where: { stripeSubscriptionId: subId },
            data: { status: "past_due" },
          });

          console.log("❌ Payment failed, subscription:", subId);
          break;
        }
        break;
      }

      default:
        console.log("🤷‍♂️ Unhandled event type:", eventType);
    }
  } catch (err) {
    console.error("❌ Error in webhook handler:", err);
    return new NextResponse("Webhook handler error", { status: 500 });
  }

  revalidatePath("/", "layout");
  return new NextResponse(null, { status: 200 });
}

// helper function
function getSafeCurrentPeriodEnd(subscription: Stripe.Subscription): Date {
  const item = subscription.items?.data?.[0];

  // Prefer the item's period end
  const end = item?.current_period_end;
  const start = item?.current_period_start;

  if (typeof end === "number" && !isNaN(end)) {
    return new Date(end * 1000);
  }

  if (typeof start === "number" && !isNaN(start)) {
    const fallback = new Date(start * 1000);
    fallback.setMonth(fallback.getMonth() + 1);
    console.warn("⚠️ Fallback: used start date + 1 month");
    return fallback;
  }

  throw new Error(
    "❌ Unable to determine currentPeriodEnd — both start and end are invalid"
  );
}
