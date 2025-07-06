import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

// Helper function to ensure we always return valid numbers
const ensureNumber = (value: number | null | undefined): number => {
	const num = Number(value ?? 0);
	return Number.isNaN(num) || !Number.isFinite(num) ? 0 : num;
};

const recordUsageSchema = z.object({
	apiKeyId: z.string(),
	provider: z.string(),
	model: z.string(),
	requestType: z.string(),
	inputTokens: z.number().default(0),
	outputTokens: z.number().default(0),
	totalTokens: z.number().default(0),
	cost: z.number().default(0),
	requestCount: z.number().default(1),
	metadata: z.record(z.any()).optional(),
});

export const usageRouter = createTRPCRouter({
	// Record API usage
	record: protectedProcedure
		.input(recordUsageSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				// Verify the API key belongs to the user
				const apiKey = await ctx.db.apiKey.findFirst({
					where: {
						id: input.apiKeyId,
						userId,
						status: "active",
					},
				});

				if (!apiKey) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "Invalid API key",
					});
				}

				// Record the usage
				const usage = await ctx.db.apiUsage.create({
					data: {
						apiKeyId: input.apiKeyId,
						projectId: apiKey.projectId,
						provider: input.provider,
						model: input.model,
						requestType: input.requestType,
						inputTokens: input.inputTokens,
						outputTokens: input.outputTokens,
						totalTokens: input.totalTokens,
						cost: input.cost,
						requestCount: input.requestCount,
						metadata: input.metadata,
					},
				});

				// Update the API key's last used timestamp
				await ctx.db.apiKey.update({
					where: { id: input.apiKeyId },
					data: { lastUsedAt: new Date() },
				});

				return usage;
			} catch (error) {
				console.error("Error recording usage:", error);
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

	// Get usage analytics for a project
	getProjectAnalytics: protectedProcedure
		.input(
			z.object({
				projectId: z.string(),
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

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
							inputTokenCost: model.inputTokenCost,
							outputTokenCost: model.outputTokenCost,
						});
					});
					providerModelMap.set(provider.name, modelMap);
				});

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
					provider: string;
					model: string;
					inputTokens: number;
					outputTokens: number;
				}) => {
					let maxCost = 0;

					// Check cost with each provider's equivalent model
					for (const [providerName, models] of providerModelMap.entries()) {
						if (providerName === usage.provider) continue; // Skip current provider

						// Try to find the exact model or a similar one
						const modelPricing =
							models.get(usage.model) || models.values().next().value;
						if (modelPricing) {
							const cost =
								(usage.inputTokens * modelPricing.inputTokenCost) / 1000000 +
								(usage.outputTokens * modelPricing.outputTokenCost) / 1000000;
							maxCost = Math.max(maxCost, cost);
						}
					}

					return maxCost;
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

				return {
					totalSpend,
					totalTokens: ensureNumber(totalMetrics._sum.totalTokens),
					totalRequests: ensureNumber(totalMetrics._sum.requestCount),
					totalApiCalls: ensureNumber(totalMetrics._count.id),
					totalEstimatedSingleProviderCost,
					totalSavings,
					totalSavingsPercentage,
					providerBreakdown: providerBreakdownWithComparison,
					requestTypeBreakdown: requestTypeUsage.map((usage) => ({
						type: usage.requestType,
						spend: ensureNumber(usage._sum.cost),
						tokens: ensureNumber(usage._sum.totalTokens),
						requests: ensureNumber(usage._sum.requestCount),
						calls: ensureNumber(usage._count.id),
					})),
					dailyTrends: dailyUsage.map((usage) => ({
						date: usage.timestamp,
						spend: ensureNumber(usage._sum.cost),
						tokens: ensureNumber(usage._sum.totalTokens),
						requests: ensureNumber(usage._sum.requestCount),
					})),
				};
			} catch (error) {
				console.error("Error fetching project analytics:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch project analytics",
				});
			}
		}),

	// Get usage analytics for a user across all projects
	getUserAnalytics: protectedProcedure
		.input(
			z.object({
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

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
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch user analytics",
				});
			}
		}),
});
