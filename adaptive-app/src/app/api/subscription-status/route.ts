import { NextResponse } from "next/server";
import { db } from "@/server/db";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");

  if (!userId) {
    return NextResponse.json({ error: "User ID is required" }, { status: 400 });
  }

  try {
    // Query the subscription table for the user's subscription
    const subscription = await db.subscription.findUnique({
      where: { userId },
    });

    // Check if the subscription exists and is active
    const isSubscribed = !!subscription && subscription.status === "active";

    return NextResponse.json({ isSubscribed });
  } catch (error) {
    console.error("Error fetching subscription status:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
