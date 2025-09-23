import { TRPCError } from "@trpc/server";
import { calculateCreditCost, deductCredits } from "@/lib/credits";
import {
	applyCacheTierDiscount,
	calculateProviderCost,
	checkSufficientCredits,
	getOrganizationId,
	getProviderModel,
	validateApiKey,
} from "@/lib/server/api-key-utils";
import { hashApiKey } from "@/lib/server/usage-utils";
import { invalidateAnalyticsCache } from "@/lib/shared/cache";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	createUsageMetadata,
	recordApiUsageInputSchema,
	recordErrorInputSchema,
} from "@/types/usage";

/**
 * Usage router for recording API usage and errors
 */
export const usageRouter = createTRPCRouter({
	// Record API usage for chat completions
	recordApiUsage: publicProcedure
		.input(recordApiUsageInputSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Validate API key and get organization info
				const apiKey = await validateApiKey(ctx.db, input.apiKey);
				const organizationId = getOrganizationId(apiKey);

				// Find provider model for cost calculation
				const providerModel = await getProviderModel(
					ctx.db,
					input.provider,
					input.model,
				);

				// Calculate costs
				const providerCost = calculateProviderCost(
					providerModel,
					input.usage.promptTokens,
					input.usage.completionTokens,
				);

				const baseCreditCost = calculateCreditCost(
					input.usage.promptTokens,
					input.usage.completionTokens,
				);
				const creditCost = applyCacheTierDiscount(
					baseCreditCost,
					input.cacheTier,
				);

				// Check credit balance before processing
				console.log("🔍 Checking credit balance before API usage.");
				await checkSufficientCredits(organizationId, creditCost);

				// Record the usage
				const usage = await ctx.db.apiUsage.create({
					data: {
						apiKeyId: apiKey.id,
						projectId: apiKey.projectId,
						clusterId: input.clusterId,
						provider: input.provider,
						model: input.model,
						requestType: "chat",
						inputTokens: input.usage.promptTokens,
						outputTokens: input.usage.completionTokens,
						totalTokens: input.usage.totalTokens,
						cost: providerCost,
						creditCost,
						requestCount: input.requestCount,
						metadata: createUsageMetadata(input, apiKey),
					},
				});

				// Handle credit deduction (only if cost > 0)
				const shouldDeductCredits = creditCost > 0;
				if (shouldDeductCredits) {
					console.log("💸 Deducting credits for API usage.");
					await deductCredits({
						organizationId,
						userId: apiKey.userId,
						amount: creditCost,
						description: `API usage: ${input.usage.promptTokens} input + ${input.usage.completionTokens} output tokens`,
						metadata: {
							provider: input.provider,
							model: input.model,
							inputTokens: input.usage.promptTokens,
							outputTokens: input.usage.completionTokens,
							duration: input.duration,
						},
						apiKeyId: apiKey.id,
						apiUsageId: usage.id,
					});
				} else {
					console.log(
						"🎯 No credit deduction needed - cached response or zero cost.",
					);
				}

				// Update API key last used timestamp and invalidate cache
				await Promise.all([
					ctx.db.apiKey.update({
						where: { id: apiKey.id },
						data: { lastUsedAt: new Date() },
					}),
					invalidateAnalyticsCache(
						apiKey.userId,
						apiKey.projectId || undefined,
					),
				]);

				console.log("✅ API usage recorded successfully.");

				return {
					success: true,
					usage,
					creditTransaction: shouldDeductCredits
						? {
								amount: creditCost,
								processed: true,
							}
						: {
								amount: 0,
								processed: false,
								reason: "No cost - cached response",
							},
				};
			} catch (error) {
				console.error("Failed to record API usage:", error);

				if (error instanceof TRPCError) {
					throw error;
				}

				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message:
						error instanceof Error ? error.message : "Failed to record usage",
					cause: error,
				});
			}
		}),

	// Record API errors
	recordError: publicProcedure
		.input(recordErrorInputSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Hash the provided API key to compare with stored hash
				const keyHash = hashApiKey(input.apiKey);

				// Find the API key in the database by the key hash
				const apiKey = await ctx.db.apiKey.findFirst({
					where: {
						keyHash,
						status: "active",
					},
					include: {
						project: {
							include: {
								organization: true,
							},
						},
					},
				});

				if (!apiKey) {
					// For error recording, still try to record even with invalid key
					console.warn("Invalid API key for error recording:", input.apiKey);
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "Invalid API key",
					});
				}

				// Record the error as usage with 0 tokens
				const usage = await ctx.db.apiUsage.create({
					data: {
						apiKeyId: apiKey.id,
						projectId: apiKey.projectId,
						provider: input.provider || "openai",
						model: input.model || "unknown",
						requestType: "chat",
						inputTokens: 0,
						outputTokens: 0,
						totalTokens: 0,
						cost: 0,
						requestCount: 1,
						metadata: {
							error: input.error,
							errorOnly: true,
							timestamp: input.timestamp,
							userId: apiKey.userId,
						},
					},
				});

				// Invalidate analytics cache
				await invalidateAnalyticsCache(
					apiKey.userId,
					apiKey.projectId || undefined,
				);

				return { success: true, usage };
			} catch (error) {
				console.error("Failed to record API error:", {
					error: error instanceof Error ? error.message : String(error),
					stack: error instanceof Error ? error.stack : undefined,
					input,
				});
				return { success: false, error: "Failed to record error" };
			}
		}),
});
