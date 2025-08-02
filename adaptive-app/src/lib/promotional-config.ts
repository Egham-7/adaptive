/**
 * Promotional credit configuration
 * Easy to modify promotional credit settings
 */

export const PROMOTIONAL_CONFIG = {
	// Amount of free credits to give to new users (in USD)
	FREE_CREDIT_AMOUNT: 1.0,

	// Maximum number of users who can receive promotional credits
	MAX_PROMOTIONAL_USERS: 100,

	// Description for promotional credit transactions
	DESCRIPTION: "Welcome bonus - $1 free credit",

	// Enable/disable promotional credits
	ENABLED: true,
} as const;

/**
 * Helper function to check if promotional credits are still available globally
 */
export async function getPromotionalCreditStats() {
	const { db } = await import("@/server/db");

	const promotionalCount = await db.creditTransaction.count({
		where: {
			type: "promotional",
			metadata: {
				path: ["promotionType"],
				equals: "new_user_bonus",
			},
		},
	});

	return {
		used: promotionalCount,
		remaining: Math.max(
			0,
			PROMOTIONAL_CONFIG.MAX_PROMOTIONAL_USERS - promotionalCount,
		),
		available:
			promotionalCount < PROMOTIONAL_CONFIG.MAX_PROMOTIONAL_USERS &&
			PROMOTIONAL_CONFIG.ENABLED,
	};
}

/**
 * Check if a specific user has already received promotional credits
 */
export async function hasUserReceivedPromotionalCredits(
	userId: string,
): Promise<boolean> {
	const { db } = await import("@/server/db");

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

	return !!existingPromotion;
}
