import type { PrismaClient } from "@prisma/client";

export async function isUserSubscribed(
	db: PrismaClient,
	userId: string,
): Promise<boolean> {
	const subscription = await db.subscription.findFirst({
		where: {
			userId: userId,
			status: "active",
			// Add any other conditions for valid subscriptions
		},
	});

	return !!subscription;
}
