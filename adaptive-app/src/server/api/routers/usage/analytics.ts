import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { withCache } from "@/lib/shared/cache";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import { ensureNumber, providerEnum } from "./shared-utils";

export const analyticsRouter = createTRPCRouter({
	// Get usage analytics for a project
	getProjectAnalytics: protectedProcedure
		.input(
			z.object({
				projectId: z.string(),
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z.enum(providerEnum).optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			const cacheKey = `project-analytics:${userId}:${
				input.projectId
			}:${JSON.stringify(input)}`;

			return withCache(cacheKey, async () => {
				try {
					// Ensure userId is available
					if (!userId) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "User ID not found in context",
						});
					}
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

					// Use proper UTC dates to avoid timezone issues
					const now = new Date();
					const endDate = input.endDate ?? now;
					const startDate =
						input.startDate ??
						new Date(endDate.getTime() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
					// Normalize to UTC day boundaries: [startUtc, endUtcExclusive)
					const startUtc = new Date(
						Date.UTC(
							startDate.getUTCFullYear(),
							startDate.getUTCMonth(),
							startDate.getUTCDate(),
						),
					);
					const endUtcExclusive = new Date(
						Date.UTC(
							endDate.getUTCFullYear(),
							endDate.getUTCMonth(),
							endDate.getUTCDate() + 1,
						),
					);

					const whereClause = {
						projectId: input.projectId,
						timestamp: {
							gte: startUtc,
							lt: endUtcExclusive,
						},
						...(input.provider && { provider: input.provider }),
					};

					// Zod schema for aggregate result
					const aggregateSchema = z.object({
						_sum: z.object({
							totalTokens: z.number().nullable(),
							cost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
							creditCost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
							requestCount: z.number().nullable(),
						}),
						_count: z.object({
							id: z.number().nullable(),
						}),
					});

					// Get total metrics
					const aggregateResult = await ctx.db.apiUsage.aggregate({
						where: whereClause,
						_sum: {
							totalTokens: true,
							cost: true,
							creditCost: true,
							requestCount: true,
						},
						_count: {
							id: true,
						},
					});

					const totalMetrics = aggregateSchema.parse(aggregateResult);

					const requestTypeUsageSchema = z.object({
						requestType: z.string(),
						_sum: z.object({
							totalTokens: z.number().nullable(),
							cost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
							requestCount: z.number().nullable(),
						}),
						_count: z.object({
							id: z.number(),
						}),
					});

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

					// Get daily usage trends - group by UTC day; make provider filter optional
					type DailyUsageRow = {
						date: Date;
						total_tokens: bigint | null;
						input_tokens: bigint | null;
						output_tokens: bigint | null;
						cost: string | null; // Decimal from Prisma is returned as string from raw queries
						credit_cost: string | null;
						request_count: bigint | null;
					};
					// Get daily usage trends - use simpler DATE function for broader compatibility
					let dailyUsageRaw: DailyUsageRow[];
					if (input.provider) {
						dailyUsageRaw = await ctx.db.$queryRaw<DailyUsageRow[]>`
							SELECT 
								DATE("timestamp") as date,
								SUM("totalTokens") as total_tokens,
								SUM("inputTokens") as input_tokens,
								SUM("outputTokens") as output_tokens,
								SUM(cost) as cost,
								SUM("creditCost") as credit_cost,
								SUM("requestCount") as request_count
							FROM "ApiUsage"
							WHERE 
								"projectId" = ${input.projectId}
								AND "timestamp" >= ${startUtc}
								AND "timestamp" < ${endUtcExclusive}
								AND provider = ${input.provider}
							GROUP BY DATE("timestamp")
							ORDER BY DATE("timestamp")
						`;
					} else {
						dailyUsageRaw = await ctx.db.$queryRaw<DailyUsageRow[]>`
							SELECT 
								DATE("timestamp") as date,
								SUM("totalTokens") as total_tokens,
								SUM("inputTokens") as input_tokens,
								SUM("outputTokens") as output_tokens,
								SUM(cost) as cost,
								SUM("creditCost") as credit_cost,
								SUM("requestCount") as request_count
							FROM "ApiUsage"
							WHERE 
								"projectId" = ${input.projectId}
								AND "timestamp" >= ${startUtc}
								AND "timestamp" < ${endUtcExclusive}
							GROUP BY DATE("timestamp")
							ORDER BY DATE("timestamp")
						`;
					}

					const dailyUsage = dailyUsageRaw.map((row) => ({
						timestamp: row.date,
						_sum: {
							totalTokens: row.total_tokens ? Number(row.total_tokens) : null,
							inputTokens: row.input_tokens ? Number(row.input_tokens) : null,
							outputTokens: row.output_tokens
								? Number(row.output_tokens)
								: null,
							cost: row.cost ? Number(row.cost) : 0,
							creditCost: row.credit_cost ? Number(row.credit_cost) : 0,
							requestCount: row.request_count
								? Number(row.request_count)
								: null,
						},
					}));

					// Calculate comparison costs using database provider pricing
					const totalSpend = ensureNumber(totalMetrics._sum.creditCost); // Use creditCost for customer spending

					// Get all providers with their pricing data
					const providers = await ctx.db.provider.findMany({
						where: {},
						include: {
							models: {
								where: {},
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

					// Calculate what it would have cost if all usage went through a specific provider/model
					// This compares Adaptive's cost to direct provider costs
					const calculateModelCostFromProvider = (
						targetModelName: string,
						targetProvider: string,
					) => {
						const targetProviderModels = providerModelMap.get(targetProvider);
						if (!targetProviderModels) return 0;

						const modelPricing = targetProviderModels.get(targetModelName);
						if (!modelPricing) return 0;

						// Apply target provider's pricing to ALL actual usage tokens
						return detailedUsage.reduce((sum, usage) => {
							// Apply target pricing to all token usage regardless of source model
							return (
								sum +
								((usage.inputTokens * modelPricing.inputTokenCost) / 1000000 +
									(usage.outputTokens * modelPricing.outputTokenCost) / 1000000)
							);
						}, 0);
					};

					// Get all unique model-provider combinations
					const modelProviderCombinations: Array<{
						model: string;
						provider: string;
						pricing: { inputTokenCost: number; outputTokenCost: number };
					}> = [];

					for (const [providerName, models] of providerModelMap.entries()) {
						for (const [modelName, pricing] of models.entries()) {
							modelProviderCombinations.push({
								model: modelName,
								provider: providerName,
								pricing,
							});
						}
					}

					const modelProviderBreakdown = modelProviderCombinations.map(
						({ model, provider, pricing }) => {
							const estimatedCost = calculateModelCostFromProvider(
								model,
								provider,
							);
							const savings = Math.max(0, estimatedCost - totalSpend);
							const savingsPercentage =
								estimatedCost > 0 ? (savings / estimatedCost) * 100 : 0;

							return {
								model,
								provider,
								estimatedCost,
								savings,
								savingsPercentage,
								pricing: {
									inputCost: pricing.inputTokenCost,
									outputCost: pricing.outputTokenCost,
								},
							};
						},
					);

					// Calculate total comparison cost (use average of all combinations)
					const totalEstimatedCost =
						modelProviderBreakdown.length > 0
							? modelProviderBreakdown.reduce(
									(sum, combo) => sum + combo.estimatedCost,
									0,
								) / modelProviderBreakdown.length
							: 0;

					const totalSavings = Math.max(0, totalEstimatedCost - totalSpend);
					const totalSavingsPercentage =
						totalEstimatedCost > 0
							? (totalSavings / totalEstimatedCost) * 100
							: 0;

					// Calculate error rate data - find all entries where metadata.error exists
					const errorUsage = await ctx.db.apiUsage.findMany({
						where: {
							...whereClause,
							metadata: {
								path: ["error"],
								not: "null",
							},
						},
						select: {
							timestamp: true,
						},
					});

					const totalCalls = ensureNumber(totalMetrics._count.id);
					const errorCount = errorUsage.length;
					const errorRate =
						totalCalls > 0 ? (errorCount / totalCalls) * 100 : 0;

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
						totalEstimatedCost,
						totalSavings,
						totalSavingsPercentage,
						errorRate,
						errorCount,
						modelProviderBreakdown,
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
								spend: ensureNumber(usage._sum.creditCost), // ← Use creditCost for customer spending
								providerCost: ensureNumber(usage._sum.cost), // ← Keep provider cost for admin
								tokens: ensureNumber(usage._sum.totalTokens),
								inputTokens: ensureNumber(usage._sum.inputTokens), // ← Add input tokens
								outputTokens: ensureNumber(usage._sum.outputTokens), // ← Add output tokens
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
			});
		}),

	// Get usage analytics for a user across all projects
	getUserAnalytics: protectedProcedure
		.input(
			z.object({
				startDate: z.date().optional(),
				endDate: z.date().optional(),
				provider: z.enum(providerEnum).optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;
			const cacheKey = `user-analytics:${userId}:${JSON.stringify(input)}`;

			return withCache(cacheKey, async () => {
				try {
					// Use proper UTC dates to avoid timezone issues
					const now = new Date();
					const endDate = input.endDate ?? now;
					const startDate =
						input.startDate ??
						new Date(endDate.getTime() - 30 * 24 * 60 * 60 * 1000);
					const startUtc = new Date(
						Date.UTC(
							startDate.getUTCFullYear(),
							startDate.getUTCMonth(),
							startDate.getUTCDate(),
						),
					);
					const endUtcExclusive = new Date(
						Date.UTC(
							endDate.getUTCFullYear(),
							endDate.getUTCMonth(),
							endDate.getUTCDate() + 1,
						),
					);

					const whereClause = {
						timestamp: { gte: startUtc, lt: endUtcExclusive },
						...(input.provider && { provider: input.provider }),
						OR: [
							{ apiKey: { userId } },
							{
								apiKeyId: null,
								metadata: { path: ["userId"], equals: userId },
							},
						],
					};

					// Zod schema for aggregate result
					const aggregateSchema = z.object({
						_sum: z.object({
							totalTokens: z.number().nullable(),
							cost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
							creditCost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
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
								creditCost: true,
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
							cost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
							creditCost: z
								.any()
								.nullable()
								.transform((val) => (val ? Number(val) : 0)),
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
								creditCost: true,
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
						totalSpend: ensureNumber(totalMetrics._sum.creditCost),
						totalTokens: ensureNumber(totalMetrics._sum.totalTokens),
						totalRequests: ensureNumber(totalMetrics._sum.requestCount),
						totalApiCalls: ensureNumber(totalMetrics._count.id),
						projectBreakdown: projectUsage.map((usage) => {
							const project = projects.find((p) => p.id === usage.projectId);
							return {
								projectId: usage.projectId,
								projectName: project?.name || "Unknown Project",
								spend: ensureNumber(usage._sum.creditCost),
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
			});
		}),
});
