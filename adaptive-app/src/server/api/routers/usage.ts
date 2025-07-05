import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

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
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to record usage",
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

				return {
					totalSpend: totalMetrics._sum.cost ?? 0,
					totalTokens: totalMetrics._sum.totalTokens ?? 0,
					totalRequests: totalMetrics._sum.requestCount ?? 0,
					totalApiCalls: totalMetrics._count.id ?? 0,
					providerBreakdown: providerUsage.map((usage) => ({
						provider: usage.provider,
						spend: usage._sum.cost ?? 0,
						tokens: usage._sum.totalTokens ?? 0,
						requests: usage._sum.requestCount ?? 0,
						calls: usage._count.id ?? 0,
					})),
					requestTypeBreakdown: requestTypeUsage.map((usage) => ({
						type: usage.requestType,
						spend: usage._sum.cost ?? 0,
						tokens: usage._sum.totalTokens ?? 0,
						requests: usage._sum.requestCount ?? 0,
						calls: usage._count.id ?? 0,
					})),
					dailyTrends: dailyUsage.map((usage) => ({
						date: usage.timestamp,
						spend: usage._sum.cost ?? 0,
						tokens: usage._sum.totalTokens ?? 0,
						requests: usage._sum.requestCount ?? 0,
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
					totalSpend: totalMetrics._sum.cost ?? 0,
					totalTokens: totalMetrics._sum.totalTokens ?? 0,
					totalRequests: totalMetrics._sum.requestCount ?? 0,
					totalApiCalls: totalMetrics._count.id ?? 0,
					projectBreakdown: projectUsage.map((usage) => {
						const project = projects.find((p) => p.id === usage.projectId);
						return {
							projectId: usage.projectId,
							projectName: project?.name || "Unknown Project",
							spend: usage._sum.cost ?? 0,
							tokens: usage._sum.totalTokens ?? 0,
							requests: usage._sum.requestCount ?? 0,
							calls: usage._count.id ?? 0,
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
