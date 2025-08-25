import { createTRPCRouter } from "@/server/api/trpc";
import { analyticsRouter } from "./usage/analytics";
import { creditOperationsRouter } from "./usage/credit-operations";
import { usageRecordingRouter } from "./usage/usage-recording";

/**
 * Combined usage router that merges all usage-related functionality:
 * - Credit operations (pre-flight checks)
 * - Usage recording (API usage tracking and error logging)
 * - Analytics (project and user analytics with cost comparisons)
 */
export const usageRouter = createTRPCRouter({
	// Credit operations
	checkCreditsBeforeUsage: creditOperationsRouter.checkCreditsBeforeUsage,

	// Usage recording
	recordApiUsage: usageRecordingRouter.recordApiUsage,
	recordError: usageRecordingRouter.recordError,

	// Analytics
	getProjectAnalytics: analyticsRouter.getProjectAnalytics,
	getUserAnalytics: analyticsRouter.getUserAnalytics,
});
