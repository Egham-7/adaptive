import crypto from "node:crypto";
import { TRPCError } from "@trpc/server";
import { Prisma } from "prisma/generated";
import { z } from "zod";
import {
	createTRPCRouter,
	protectedProcedure,
	publicProcedure,
} from "@/server/api/trpc";

// Helper function to ensure we always return valid numbers
const ensureNumber = (value: number | null | undefined): number => {
	const num = Number(value ?? 0);
	return Number.isNaN(num) || !Number.isFinite(num) ? 0 : num;
};

export const usageRouter = createTRPCRouter({
	// Record API usage for chat completions
	recordApiUsage: publicProcedure
		.input(
			z.object({
				apiKey: z.string(),
				provider: z
					.enum([
						"openai",
						"anthropic",
						"gemini",
						"groq",
						"deepseek",
						"huggingface",
						"grok",
					])
					.nullable(),
				model: z.string().nullable(),
				usage: z.object({
					promptTokens: z.number(),
					completionTokens: z.number(),
					totalTokens: z.number(),
				}),
				duration: z.number(),
				timestamp: z.date(),
				requestCount: z.number().default(1),
				metadata: z.record(z.any()).optional(),
				error: z.string().optional(), // Add error field for failed requests
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Hash the provided API key to compare with stored hash
				const keyHash = crypto
					.createHash("sha256")
					.update(input.apiKey)
					.digest("hex");

				// Find the API key in the database by the key hash
				const apiKey = await ctx.db.apiKey.findFirst({
					where: {
						keyHash,
						status: "active",
					},
				});

				if (!apiKey) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "Invalid API key",
					});
				}

				// Calculate cost using provider cost table
				const providerModel =
					!input.provider || !input.model
						? null
						: await ctx.db.providerModel
								.findFirst({
									where: {
										name: input.model,
										provider: { name: input.provider, isActive: true },
										isActive: true,
									},
									select: { inputTokenCost: true, outputTokenCost: true },
								})
								.catch(() => null);

				const calculatedCost = providerModel
					? (input.usage.promptTokens *
							providerModel.inputTokenCost.toNumber() +
							input.usage.completionTokens *
								providerModel.outputTokenCost.toNumber()) /
						1000000
					: 0;

				// Record the usage
				const usage = await ctx.db.apiUsage.create({
					data: {
						apiKeyId: apiKey.id,
						projectId: apiKey.projectId,
						provider: input.provider,
						model: input.model,
						requestType: "chat", // Default to chat for chat completions
						inputTokens: input.usage.promptTokens,
						outputTokens: input.usage.completionTokens,
						totalTokens: input.usage.totalTokens,
						cost: calculatedCost,
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

				// Update the API key's last used timestamp
				await ctx.db.apiKey.update({
					where: { id: apiKey.id },
					data: { lastUsedAt: new Date() },
				});

				return { success: true, usage };
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
				provider: z
					.enum([
						"openai",
						"anthropic",
						"gemini",
						"groq",
						"deepseek",
						"huggingface",
						"grok",
					])
					.optional(),
				model: z.string().optional(),
				error: z.string(),
				timestamp: z.date(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Hash the provided API key to compare with stored hash
				const keyHash = crypto
					.createHash("sha256")
					.update(input.apiKey)
					.digest("hex");

				// Find the API key in the database by the key hash
				const apiKey = await ctx.db.apiKey.findFirst({
					where: {
						keyHash,
						status: "active",
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

	// Get usage analytics for a project
	getProjectAnalytics: protectedProcedure
		.input(
			z.object({
				projectId: z.string(),
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z
					.enum([
						"openai",
						"anthropic",
						"gemini",
						"groq",
						"deepseek",
						"huggingface",
					])
					.optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			try {
				// Verify user has access to the project
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.projectId,
						OR: [
							{ members: { some: { userId } } },
							{ organization: { ownerId: userId } },
							{ organization: { members: { some: { userId } } } },
						],
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have access to this project",
					});
				}

				const startDate =
					input.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
				const endDate = input.endDate || new Date();

				const whereClause = {
					projectId: input.projectId,
					timestamp: {
						gte: startDate,
						lte: endDate,
					},
					...(input.provider && { provider: input.provider }),
				};

				// Zod schema for aggregate result
				const aggregateSchema = z.object({
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
					_count: z.object({
						id: z.number().nullable(),
					}),
				});

				// Get total metrics
				const totalMetrics = aggregateSchema.parse(
					await ctx.db.apiUsage.aggregate({
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					}),
				);

				// Zod schemas for groupBy results
				const providerUsageSchema = z.object({
					provider: z.string(),
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
					_count: z.object({
						id: z.number(),
					}),
				});
				const requestTypeUsageSchema = z.object({
					requestType: z.string(),
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
					_count: z.object({
						id: z.number(),
					}),
				});
				const dailyUsageSchema = z.object({
					timestamp: z.date(),
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
				});

				// Get usage by provider
				const providerUsage = providerUsageSchema.array().parse(
					await ctx.db.apiUsage.groupBy({
						by: ["provider"],
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					}),
				);

				// Get usage by request type
				const requestTypeUsage = requestTypeUsageSchema.array().parse(
					await ctx.db.apiUsage.groupBy({
						by: ["requestType"],
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					}),
				);

				// Get daily usage trends
				const dailyUsage = dailyUsageSchema.array().parse(
					await ctx.db.apiUsage.groupBy({
						by: ["timestamp"],
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						orderBy: {
							timestamp: "asc",
						},
					}),
				);

				// Calculate comparison costs using database provider pricing
				const totalSpend = ensureNumber(totalMetrics._sum.cost);

				// Get all providers with their pricing data
				const providers = await ctx.db.provider.findMany({
					where: { isActive: true },
					include: {
						models: {
							where: { isActive: true },
						},
					},
				});

				// Create a map of provider models for quick lookup
				const providerModelMap = new Map<
					string,
					Map<string, { inputTokenCost: number; outputTokenCost: number }>
				>();
				providers.forEach((provider) => {
					const modelMap = new Map<
						string,
						{ inputTokenCost: number; outputTokenCost: number }
					>();
					provider.models.forEach((model) => {
						modelMap.set(model.name, {
							inputTokenCost: model.inputTokenCost.toNumber(),
							outputTokenCost: model.outputTokenCost.toNumber(),
						});
					});
					providerModelMap.set(provider.name, modelMap);
				});

				// Pre-compute maximum cost per model across all providers
				const maxCostPerModel = new Map<
					string,
					{ inputCost: number; outputCost: number }
				>();

				for (const [_providerName, models] of providerModelMap.entries()) {
					for (const [modelName, modelPricing] of models.entries()) {
						const existing = maxCostPerModel.get(modelName);
						const inputCost = Number(modelPricing.inputTokenCost);
						const outputCost = Number(modelPricing.outputTokenCost);

						if (
							!existing ||
							inputCost > existing.inputCost ||
							outputCost > existing.outputCost
						) {
							maxCostPerModel.set(modelName, {
								inputCost: Math.max(inputCost, existing?.inputCost || 0),
								outputCost: Math.max(outputCost, existing?.outputCost || 0),
							});
						}
					}
				}

				// Get detailed usage data with model information for cost calculations
				const detailedUsage = await ctx.db.apiUsage.findMany({
					where: whereClause,
					select: {
						provider: true,
						model: true,
						inputTokens: true,
						outputTokens: true,
						cost: true,
					},
				});

				// Calculate what the cost would be if using only the most expensive provider
				const calculateAlternativeProviderCost = (usage: {
					provider: string | null;
					model: string | null;
					inputTokens: number;
					outputTokens: number;
				}) => {
					if (!usage.model || !usage.provider) return 0;

					const maxCost = maxCostPerModel.get(usage.model);
					if (!maxCost) return 0;

					return (
						(usage.inputTokens * maxCost.inputCost) / 1000000 +
						(usage.outputTokens * maxCost.outputCost) / 1000000
					);
				};

				// Calculate savings for each provider
				const providerBreakdownWithComparison = providerUsage.map((usage) => {
					const spend = ensureNumber(usage._sum.cost);

					// Calculate what this provider's usage would cost with the most expensive alternative
					const relevantUsage = detailedUsage.filter(
						(u) => u.provider === usage.provider,
					);
					const estimatedAlternativeCost = relevantUsage.reduce((sum, u) => {
						return sum + calculateAlternativeProviderCost(u);
					}, 0);

					const savings = Math.max(0, estimatedAlternativeCost - spend);
					const savingsPercentage =
						estimatedAlternativeCost > 0
							? (savings / estimatedAlternativeCost) * 100
							: 0;

					return {
						provider: usage.provider,
						spend,
						tokens: ensureNumber(usage._sum.totalTokens),
						requests: ensureNumber(usage._sum.requestCount),
						calls: ensureNumber(usage._count.id),
						estimatedSingleProviderCost: estimatedAlternativeCost,
						savings,
						savingsPercentage,
					};
				});

				// Calculate total comparison cost across all providers
				const totalEstimatedSingleProviderCost =
					providerBreakdownWithComparison.reduce(
						(sum, provider) => sum + provider.estimatedSingleProviderCost,
						0,
					);

				const totalSavings = Math.max(
					0,
					totalEstimatedSingleProviderCost - totalSpend,
				);
				const totalSavingsPercentage =
					totalEstimatedSingleProviderCost > 0
						? (totalSavings / totalEstimatedSingleProviderCost) * 100
						: 0;

				// Calculate error rate data - find all entries where metadata.error exists
				const errorUsage = await ctx.db.apiUsage.findMany({
					where: {
						...whereClause,
						metadata: {
							path: ["error"],
							not: Prisma.JsonNull,
						},
					},
					select: {
						timestamp: true,
					},
				});

				const totalCalls = ensureNumber(totalMetrics._count.id);
				const errorCount = errorUsage.length;
				const errorRate = totalCalls > 0 ? (errorCount / totalCalls) * 100 : 0;

				// Group errors by day for trend analysis
				const errorsByDay = errorUsage.reduce(
					(acc, usage) => {
						const dateKey = usage.timestamp.toISOString().split("T")[0];
						if (dateKey) {
							acc[dateKey] = (acc[dateKey] || 0) + 1;
						}
						return acc;
					},
					{} as Record<string, number>,
				);

				return {
					totalSpend,
					totalTokens: ensureNumber(totalMetrics._sum.totalTokens),
					totalRequests: ensureNumber(totalMetrics._sum.requestCount),
					totalApiCalls: totalCalls,
					totalEstimatedSingleProviderCost,
					totalSavings,
					totalSavingsPercentage,
					errorRate,
					errorCount,
					providerBreakdown: providerBreakdownWithComparison,
					requestTypeBreakdown: requestTypeUsage.map((usage) => ({
						type: usage.requestType,
						spend: ensureNumber(usage._sum.cost),
						tokens: ensureNumber(usage._sum.totalTokens),
						requests: ensureNumber(usage._sum.requestCount),
						calls: ensureNumber(usage._count.id),
					})),
					dailyTrends: dailyUsage.map((usage) => {
						const dateKey = usage.timestamp.toISOString().split("T")[0];
						return {
							date: usage.timestamp,
							spend: ensureNumber(usage._sum.cost),
							tokens: ensureNumber(usage._sum.totalTokens),
							requests: ensureNumber(usage._sum.requestCount),
							errorCount: dateKey ? errorsByDay[dateKey] || 0 : 0,
						};
					}),
				};
			} catch (error) {
				console.error("Error fetching project analytics:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message:
						error instanceof Error
							? error.message
							: "Failed to fetch project analytics",
					cause: error,
				});
			}
		}),

	// Get usage analytics for a user across all projects
	getUserAnalytics: protectedProcedure
		.input(
			z.object({
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z
					.enum([
						"openai",
						"anthropic",
						"gemini",
						"groq",
						"deepseek",
						"huggingface",
					])
					.optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			try {
				const startDate =
					input.startDate || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
				const endDate = input.endDate || new Date();

				const whereClause = {
					apiKey: { userId },
					timestamp: {
						gte: startDate,
						lte: endDate,
					},
					...(input.provider && { provider: input.provider }),
				};

				// Zod schema for aggregate result
				const aggregateSchema = z.object({
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
					_count: z.object({
						id: z.number().nullable(),
					}),
				});

				// Get total metrics
				const totalMetrics = aggregateSchema.parse(
					await ctx.db.apiUsage.aggregate({
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					}),
				);

				const projectUsageSchema = z.object({
					projectId: z.string(),
					_sum: z.object({
						totalTokens: z.number().nullable(),
						cost: z.number().nullable(),
						requestCount: z.number().nullable(),
					}),
					_count: z.object({
						id: z.number(),
					}),
				});

				const projectUsage = projectUsageSchema.array().parse(
					await ctx.db.apiUsage.groupBy({
						by: ["projectId"],
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					}),
				);

				// Get project details for the usage
				const projectIds = projectUsage
					.map((usage) => usage.projectId)
					.filter(Boolean) as string[];
				const projects = await ctx.db.project.findMany({
					where: { id: { in: projectIds } },
					select: { id: true, name: true },
				});

				return {
					totalSpend: ensureNumber(totalMetrics._sum.cost),
					totalTokens: ensureNumber(totalMetrics._sum.totalTokens),
					totalRequests: ensureNumber(totalMetrics._sum.requestCount),
					totalApiCalls: ensureNumber(totalMetrics._count.id),
					projectBreakdown: projectUsage.map((usage) => {
						const project = projects.find((p) => p.id === usage.projectId);
						return {
							projectId: usage.projectId,
							projectName: project?.name || "Unknown Project",
							spend: ensureNumber(usage._sum.cost),
							tokens: ensureNumber(usage._sum.totalTokens),
							requests: ensureNumber(usage._sum.requestCount),
							calls: ensureNumber(usage._count.id),
						};
					}),
				};
			} catch (error) {
				console.error("Error fetching user analytics:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message:
						error instanceof Error
							? error.message
							: "Failed to fetch user analytics",
					cause: error,
				});
			}
		}),
});
