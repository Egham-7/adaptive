import { TRPCError } from "@trpc/server";
import { getDetailedPromotionalStats } from "@/lib/credits/admin";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

// Admin user IDs from environment variable
const ADMIN_USER_IDS =
	process.env.ADMIN_USER_IDS?.split(",").map((id) => id.trim()) || [];

/**
 * Middleware to check if user is admin
 */
const adminProcedure = protectedProcedure.use(async ({ ctx, next }) => {
	const userId = ctx.userId;

	if (!userId || !ADMIN_USER_IDS.includes(userId)) {
		throw new TRPCError({
			code: "FORBIDDEN",
			message: "Admin access required",
		});
	}

	return next({
		ctx: {
			...ctx,
			userId,
		},
	});
});

export const adminRouter = createTRPCRouter({
	/**
	 * Get detailed promotional credit statistics
	 * Admin only - shows all promotional credit awards
	 */
	getPromotionalStats: adminProcedure.query(async () => {
		try {
			const stats = await getDetailedPromotionalStats();
			return stats;
		} catch (error) {
			console.error("Error fetching promotional stats:", error);
			throw new TRPCError({
				code: "INTERNAL_SERVER_ERROR",
				message: "Failed to fetch promotional credit statistics",
				cause: error,
			});
		}
	}),
});
