import { z } from "zod";
import { providerEnum } from "@/types/providers";

/**
 * Schema for project analytics input
 */
export const projectAnalyticsInputSchema = z.object({
	projectId: z.string(),
	startDate: z.date().optional(),
	endDate: z.date().optional(),
	provider: z.enum(providerEnum).optional(),
});

/**
 * Schema for user analytics input
 */
export const userAnalyticsInputSchema = z.object({
	startDate: z.date().optional(),
	endDate: z.date().optional(),
	provider: z.enum(providerEnum).optional(),
});

// Type inference from schemas
export type ProjectAnalyticsInput = z.infer<typeof projectAnalyticsInputSchema>;
export type UserAnalyticsInput = z.infer<typeof userAnalyticsInputSchema>;

/**
 * Type for daily usage raw database row
 */
export type DailyUsageRow = {
	date: Date;
	total_tokens: bigint | null;
	input_tokens: bigint | null;
	output_tokens: bigint | null;
	cost: string | null; // Decimal from Prisma is returned as string from raw queries
	credit_cost: string | null;
	request_count: bigint | null;
};
