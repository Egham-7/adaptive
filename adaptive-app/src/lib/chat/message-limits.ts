import type { PrismaClient } from "@prisma/client";
import { endOfDay, startOfDay } from "date-fns";

export const DAILY_MESSAGE_LIMIT = 7;

export async function getDailyMessageCount(
	db: PrismaClient,
	userId: string,
): Promise<number> {
	const today = new Date();
	const startOfToday = startOfDay(today);
	const endOfToday = endOfDay(today);

	const messageCount = await db.message.count({
		where: {
			conversation: {
				userId: userId,
			},
			role: "user", // Only count user messages, not AI responses
			createdAt: {
				gte: startOfToday,
				lte: endOfToday,
			},
			deletedAt: null,
		},
	});

	return messageCount;
}

export async function hasReachedDailyLimit(
	db: PrismaClient,
	userId: string,
): Promise<boolean> {
	const messageCount = await getDailyMessageCount(db, userId);
	return messageCount >= DAILY_MESSAGE_LIMIT;
}

export async function getRemainingMessages(
	db: PrismaClient,
	userId: string,
): Promise<number> {
	const messageCount = await getDailyMessageCount(db, userId);
	return Math.max(0, DAILY_MESSAGE_LIMIT - messageCount);
}
