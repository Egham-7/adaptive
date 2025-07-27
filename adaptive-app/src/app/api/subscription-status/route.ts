import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { db } from "@/server/db";

export async function GET(req: Request) {
	const { userId: authenticatedUserId } = await auth(); // Authenticate the user
	const { searchParams } = new URL(req.url);
	const userId = searchParams.get("userId");

	// Ensure the user is authenticated
	if (!authenticatedUserId) {
		return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
	}

	// Ensure the user can only query their own subscription status
	if (userId !== authenticatedUserId) {
		return NextResponse.json({ error: "Forbidden" }, { status: 403 });
	}

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
			{ status: 500 },
		);
	}
}
