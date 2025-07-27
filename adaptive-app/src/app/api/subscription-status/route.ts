import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { db } from "@/server/db";

export async function GET() {
	const { userId } = await auth(); // Authenticate the user

	// Ensure the user is authenticated
	if (!userId) {
		return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
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
			{ status: 500 },
		);
	}
}
