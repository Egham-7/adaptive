/**
 * Admin utilities for promotional credit management
 */

import { PROMOTIONAL_CONFIG } from "@/lib/promotional-config";
import { db } from "@/server/db";

/**
 * Get detailed promotional credit statistics for admin dashboard
 */
export async function getDetailedPromotionalStats() {
	const transactions = await db.creditTransaction.findMany({
		where: {
			type: "promotional",
			metadata: {
				path: ["promotionType"],
				equals: "new_user_bonus",
			},
		},
		orderBy: {
			createdAt: "asc",
		},
	});

	// Get organization details separately
	const organizationIds = [
		...new Set(transactions.map((tx) => tx.organizationId)),
	];
	const organizations = await db.organization.findMany({
		where: { id: { in: organizationIds } },
		select: { id: true, name: true, ownerId: true },
	});

	const orgMap = new Map(organizations.map((org) => [org.id, org]));

	return {
		config: PROMOTIONAL_CONFIG,
		stats: {
			used: transactions.length,
			remaining: Math.max(
				0,
				PROMOTIONAL_CONFIG.MAX_PROMOTIONAL_USERS - transactions.length,
			),
			available:
				transactions.length < PROMOTIONAL_CONFIG.MAX_PROMOTIONAL_USERS &&
				PROMOTIONAL_CONFIG.ENABLED,
		},
		transactions: transactions.map((tx, index) => ({
			userNumber: index + 1,
			userId: tx.userId,
			organizationId: tx.organizationId,
			organizationName: orgMap.get(tx.organizationId)?.name || "Unknown",
			amount: tx.amount.toNumber(),
			awardedAt: tx.createdAt,
			metadata: tx.metadata,
		})),
	};
}

/**
 * Manually award promotional credits to a specific user (admin function)
 * Use with caution - bypasses normal limits
 */
export async function manuallyAwardPromotionalCredits(
	userId: string,
	organizationId: string,
	reason: string,
) {
	const organization = await db.organization.findUnique({
		where: { id: organizationId },
	});

	if (!organization) {
		throw new Error(`Organization ${organizationId} not found`);
	}

	if (organization.ownerId !== userId) {
		throw new Error(
			`User ${userId} does not own organization ${organizationId}`,
		);
	}

	// Check if user already has promotional credits
	const existingPromotion = await db.creditTransaction.findFirst({
		where: {
			userId,
			type: "promotional",
			metadata: {
				path: ["promotionType"],
				equals: "new_user_bonus",
			},
		},
	});

	if (existingPromotion) {
		throw new Error(`User ${userId} already received promotional credits`);
	}

	// Award credits using the existing addCredits function
	const { addCredits } = await import("@/lib/credit-utils");

	return await addCredits({
		organizationId,
		userId,
		amount: PROMOTIONAL_CONFIG.FREE_CREDIT_AMOUNT,
		type: "promotional",
		description: `Manual promotional credit: ${reason}`,
		metadata: {
			promotionType: "new_user_bonus",
			awardedAt: new Date().toISOString(),
			manual: true,
			reason,
		},
	});
}
