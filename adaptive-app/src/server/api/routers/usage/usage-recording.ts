import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { z } from "zod";
import { invalidateAnalyticsCache } from "@/lib/cache-utils";
import {
	calculateCreditCost,
	deductCredits,
	getOrganizationBalance,
	hasSufficientCredits,
} from "@/lib/credit-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import type { CacheTier } from "@/types/chat-completion";
import { cacheTierSchema } from "@/types/chat-completion";
import {
	findModelBySimilarity,
	hashApiKey,
	providerEnum,
} from "./shared-utils";

// Input validation schemas
const usageInputSchema = z.object({
	promptTokens: z.number().min(0, "Prompt tokens must be non-negative"),
	completionTokens: z.number().min(0, "Completion tokens must be non-negative"),
	totalTokens: z.number().min(0, "Total tokens must be non-negative"),
});

const recordApiUsageInputSchema = z.object({
	apiKey: z.string().min(1, "API key is required"),
	provider: z.enum(providerEnum).nullable(),
	model: z.string().nullable(),
	usage: usageInputSchema,
	duration: z.number().min(0, "Duration must be non-negative"),
	timestamp: z.date(),
	requestCount: z.number().min(1).default(1),
	clusterId: z.string().optional(),
	metadata: z.record(z.string(), z.any()).optional(),
	error: z.string().optional(),
	cacheTier: cacheTierSchema.optional(),
});

const recordErrorInputSchema = z.object({
	apiKey: z.string().min(1, "API key is required"),
	provider: z.enum(providerEnum).optional(),
	model: z.string().optional(),
	error: z.string().min(1, "Error message is required"),
	timestamp: z.date(),
});

// Type definitions
type ApiKeyWithProject = Prisma.ApiKeyGetPayload<{
	include: {
		project: {
			include: {
				organization: true;
			};
		};
	};
}>;

type UsageInput = z.infer<typeof recordApiUsageInputSchema>;

// Pure functions for business logic
function applyCacheTierDiscount(
	baseCost: number,
	cacheTier?: CacheTier,
): number {
	switch (cacheTier) {
		case "prompt_response":
			return 0; // Free for prompt_response cache hits
		case "semantic_exact":
		case "semantic_similar":
			return baseCost * 0.5; // 50% discount for semantic cache hits
		default:
			return baseCost;
	}
}

function calculateProviderCost(
	providerModel: {
		inputTokenCost: { toNumber(): number };
		outputTokenCost: { toNumber(): number };
	} | null,
	promptTokens: number,
	completionTokens: number,
): number {
	if (!providerModel) return 0;

	return (
		(promptTokens * providerModel.inputTokenCost.toNumber() +
			completionTokens * providerModel.outputTokenCost.toNumber()) /
		1_000_000
	);
}

async function validateApiKey(
	db: any,
	apiKey: string,
): Promise<ApiKeyWithProject> {
	const keyHash = hashApiKey(apiKey);

	const apiKeyRecord = await db.apiKey.findFirst({
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

	if (!apiKeyRecord) {
		throw new TRPCError({
			code: "FORBIDDEN",
			message: "Invalid API key",
		});
	}

	return apiKeyRecord;
}

function getOrganizationId(apiKey: ApiKeyWithProject): string {
	const organizationId = apiKey.project?.organizationId;

	if (!organizationId) {
		throw new TRPCError({
			code: "INTERNAL_SERVER_ERROR",
			message: "API key is not associated with an organization",
		});
	}

	return organizationId;
}

async function checkSufficientCredits(
	organizationId: string,
	creditCost: number,
): Promise<void> {
	if (creditCost <= 0) return; // Skip check for free operations

	const hasEnoughCredits = await hasSufficientCredits(
		organizationId,
		creditCost,
	);

	if (!hasEnoughCredits) {
		const currentBalance = await getOrganizationBalance(organizationId);
		throw new TRPCError({
			code: "PAYMENT_REQUIRED",
			message: `Insufficient credits. Required: $${creditCost.toFixed(4)}, Available: $${currentBalance.toFixed(4)}. Please purchase more credits.`,
		});
	}
}

function createUsageMetadata(input: UsageInput, apiKey: ApiKeyWithProject) {
	return {
		...input.metadata,
		duration: input.duration,
		timestamp: input.timestamp,
		error: input.error,
		userId: apiKey.userId,
		cacheTier: input.cacheTier,
	};
}

async function getProviderModel(
	db: any,
	provider: string | null,
	model: string | null,
): Promise<{
	inputTokenCost: { toNumber(): number };
	outputTokenCost: { toNumber(): number };
} | null> {
	if (!provider || !model) return null;

	try {
		return await findModelBySimilarity(db, model, provider);
	} catch (error) {
		console.error("Error fetching provider model:", error);
		return null;
	}
}

export const usageRecordingRouter = createTRPCRouter({
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
