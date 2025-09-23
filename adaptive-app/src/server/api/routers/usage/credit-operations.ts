import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
	calculateCreditCost,
	getOrganizationBalance,
	hasSufficientCredits,
} from "@/lib/credits";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import { hashApiKey } from "./shared-utils";

export const creditOperationsRouter = createTRPCRouter({
	// Pre-flight credit check before API usage
	checkCreditsBeforeUsage: publicProcedure
		.input(
			z.object({
				apiKey: z.string(),
				estimatedInputTokens: z
					.number()
					.min(0, "Input tokens must be non-negative"),
				estimatedOutputTokens: z
					.number()
					.min(0, "Output tokens must be non-negative"),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			// Hash the API key to find it in database
			const keyHash = hashApiKey(input.apiKey);

			// Verify API key and get organization
			const apiKey = await ctx.db.apiKey.findFirst({
				where: { keyHash },
				include: { project: { include: { organization: true } } },
			});

			if (!apiKey || !apiKey.project) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "Invalid API key",
				});
			}

			const organizationId = apiKey.project.organization.id;

			// Calculate estimated credit cost
			const estimatedCreditCost = calculateCreditCost(
				input.estimatedInputTokens,
				input.estimatedOutputTokens,
			);

			// Check if organization has sufficient credits
			const hasEnoughCredits = await hasSufficientCredits(
				organizationId,
				estimatedCreditCost,
			);

			if (!hasEnoughCredits) {
				const currentBalance = await getOrganizationBalance(organizationId);
				throw new TRPCError({
					code: "PAYMENT_REQUIRED",
					message: `Insufficient credits. Estimated cost: $${estimatedCreditCost.toFixed(
						4,
					)}, Available: $${currentBalance.toFixed(
						4,
					)}. Please purchase more credits.`,
				});
			}

			return {
				hasEnoughCredits: true,
				currentBalance: await getOrganizationBalance(organizationId),
				estimatedCost: estimatedCreditCost,
			};
		}),
});
