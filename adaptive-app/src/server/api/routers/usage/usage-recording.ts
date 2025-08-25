import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { invalidateAnalyticsCache } from "@/lib/cache-utils";
import {
	calculateCreditCost,
	deductCredits,
	hasSufficientCredits,
} from "@/lib/credit-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	findModelBySimilarity,
	hashApiKey,
	providerEnum,
} from "./shared-utils";

export const usageRecordingRouter = createTRPCRouter({
	// Record API usage for chat completions
	recordApiUsage: publicProcedure
		.input(
			z.object({
				apiKey: z.string(),
				provider: z.enum(providerEnum).nullable(),
				model: z.string().nullable(),
				usage: z.object({
					promptTokens: z.number().min(0, "Prompt tokens must be non-negative"),
					completionTokens: z
						.number()
						.min(0, "Completion tokens must be non-negative"),
					totalTokens: z.number().min(0, "Total tokens must be non-negative"),
				}),
				duration: z.number(),
				timestamp: z.date(),
				requestCount: z.number().default(1),
				clusterId: z.string().optional(), // Link usage to cluster
				metadata: z.record(z.string(), z.any()).optional(),
				error: z.string().optional(), // Add error field for failed requests
			}),
		)
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
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "Invalid API key",
					});
				}

				// Get organization ID from API key's project
				const organizationId = apiKey.project?.organizationId;
				if (!organizationId) {
					throw new TRPCError({
						code: "INTERNAL_SERVER_ERROR",
						message: "API key is not associated with an organization",
					});
				}

				const providerModel =
					!input.provider || !input.model
						? null
						: await findModelBySimilarity(
								ctx.db,
								input.model,
								input.provider,
							).catch((error) => {
								console.error("Error fetching provider model:", error);
								return null;
							});

				// Calculate provider cost (what you pay to the provider)
				const calculatedCost = providerModel
					? (input.usage.promptTokens *
							providerModel.inputTokenCost.toNumber() +
							input.usage.completionTokens *
								providerModel.outputTokenCost.toNumber()) /
						1000000
					: 0;

				console.log("Input:", input);

				// Calculate credit cost (what you charge the user)
				const creditCost = calculateCreditCost(
					input.usage.promptTokens,
					input.usage.completionTokens,
				);

				console.log("🔍 Checking credit balance before API usage:.");

				// Check if organization has sufficient credits before processing
				const hasEnoughCredits = await hasSufficientCredits(
					organizationId,
					creditCost,
				);

				if (!hasEnoughCredits) {
					const currentBalance = await getOrganizationBalance(organizationId);
					throw new TRPCError({
						code: "PAYMENT_REQUIRED",
						message: `Insufficient credits. Required: $${creditCost.toFixed(
							4,
						)}, Available: $${currentBalance.toFixed(
							4,
						)}. Please purchase more credits.`,
					});
				}

				// Record the usage
				const usage = await ctx.db.apiUsage.create({
					data: {
						apiKeyId: apiKey.id,
						projectId: apiKey.projectId,
						clusterId: input.clusterId, // Link to cluster if provided
						provider: input.provider,
						model: input.model,
						requestType: "chat", // Default to chat for chat completions
						inputTokens: input.usage.promptTokens,
						outputTokens: input.usage.completionTokens,
						totalTokens: input.usage.totalTokens,
						cost: calculatedCost, // Provider cost
						creditCost: creditCost, // User credit cost
						requestCount: input.requestCount,
						metadata: {
							...input.metadata,
							duration: input.duration,
							timestamp: input.timestamp,
							error: input.error, // Include error in metadata if present
							userId: apiKey.userId, // Get userId from the API key
						},
					},
				});

				console.log("💸 Deducting credits for API usage.");

				// Deduct credits from organization's account
				const creditTransaction = await deductCredits({
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

				// Update the API key's last used timestamp
				await ctx.db.apiKey.update({
					where: { id: apiKey.id },
					data: { lastUsedAt: new Date() },
				});

				// Invalidate analytics cache
				await invalidateAnalyticsCache(
					apiKey.userId,
					apiKey.projectId || undefined,
				);

				console.log("✅ API usage recorded and credits deducted.");

				return {
					success: true,
					usage,
					creditTransaction: {
						amount: creditTransaction.deductedAmount,
						newBalance: creditTransaction.newBalance,
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
		.input(
			z.object({
				apiKey: z.string(),
				provider: z.enum(providerEnum).optional(),
				model: z.string().optional(),
				error: z.string(),
				timestamp: z.date(),
			}),
		)
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
							userId: apiKey.userId, // Get userId from the API key
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

// Import needed for recordApiUsage function
import { getOrganizationBalance } from "@/lib/credit-utils";
