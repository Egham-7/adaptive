import crypto from "node:crypto";
import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { z } from "zod";
import { authenticateAndGetProject, getCacheKey } from "@/lib/auth-utils";
import { invalidateProjectCache, withCache } from "@/lib/cache-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	addModelSchema,
	clusterByNameParamsSchema,
	createClusterSchema,
	projectClusterParamsSchema,
	updateClusterSchema,
} from "@/types/cluster-schemas";

type LLMClusterWithModels = Prisma.LLMClusterGetPayload<{
	include: {
		models: true;
	};
}>;

export const llmClustersRouter = createTRPCRouter({
	// Get all clusters for a project
	getByProject: publicProcedure
		.input(projectClusterParamsSchema)
		.query(async ({ ctx, input }): Promise<LLMClusterWithModels[]> => {
			try {
				const auth = await authenticateAndGetProject(ctx, input);

				const cacheKey = getCacheKey(auth, input.projectId);

				return withCache(cacheKey, async () => {
					const clusters = await ctx.db.lLMCluster.findMany({
						where: {
							projectId: input.projectId,
							isActive: true,
						},
						include: {
							models: {
								where: { isActive: true },
								orderBy: { priority: "asc" },
							},
						},
						orderBy: { createdAt: "desc" },
					});

					return clusters as LLMClusterWithModels[];
				});
			} catch (error) {
				console.error("Error fetching LLM clusters:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch LLM clusters",
				});
			}
		}),

	// Get cluster by name (for backend integration)
	getByName: publicProcedure
		.input(clusterByNameParamsSchema)
		.query(async ({ ctx, input }) => {
			try {
				await authenticateAndGetProject(ctx, input);

				const cluster = await ctx.db.lLMCluster.findFirst({
					where: {
						projectId: input.projectId,
						name: input.name,
						isActive: true,
					},
					include: {
						models: {
							where: { isActive: true },
							orderBy: { priority: "asc" },
						},
					},
				});

				if (!cluster) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Cluster not found",
					});
				}

				return cluster;
			} catch (error) {
				if (error instanceof TRPCError) throw error;
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch cluster",
				});
			}
		}),

	// Get cluster by ID
	getById: publicProcedure
		.input(
			z.object({
				id: z.string(),
				apiKey: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			try {
				const cluster = await ctx.db.lLMCluster.findFirst({
					where: { id: input.id, isActive: true },
					include: {
						models: {
							where: { isActive: true },
							orderBy: { priority: "asc" },
						},
						project: true,
					},
				});

				if (!cluster) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Cluster not found",
					});
				}

				// Check access to the project
				await authenticateAndGetProject(ctx, {
					projectId: cluster.projectId,
					apiKey: input.apiKey,
				});
				return cluster;
			} catch (error) {
				if (error instanceof TRPCError) throw error;
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch cluster",
				});
			}
		}),

	// Create a new LLM cluster (atomic transaction)
	create: publicProcedure
		.input(createClusterSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Check project access
				const auth = await authenticateAndGetProject(ctx, input);

				// Atomic transaction for cluster creation
				const cluster = await ctx.db.$transaction(async (tx) => {
					// Check if cluster name already exists in project
					const existingCluster = await tx.lLMCluster.findFirst({
						where: {
							projectId: input.projectId,
							name: input.name,
						},
					});

					if (existingCluster) {
						throw new TRPCError({
							code: "CONFLICT",
							message: "Cluster name already exists in this project",
						});
					}

					// Validate all models exist
					for (const model of input.models) {
						const providerModel = await tx.providerModel.findFirst({
							where: {
								provider: { name: model.provider },
								name: model.modelName,
								isActive: true,
							},
						});

						if (!providerModel) {
							throw new TRPCError({
								code: "BAD_REQUEST",
								message: `Model ${model.modelName} from ${model.provider} not found or not available`,
							});
						}
					}

					// Create cluster
					const newCluster = await tx.lLMCluster.create({
						data: {
							projectId: input.projectId,
							name: input.name,
							description: input.description,
							fallbackEnabled: input.fallbackEnabled,
							fallbackMode: input.fallbackMode,
							enableCircuitBreaker: input.enableCircuitBreaker,
							maxRetries: input.maxRetries,
							timeoutMs: input.timeoutMs,
							costBias: input.costBias,
							complexityThreshold: input.complexityThreshold,
							tokenThreshold: input.tokenThreshold,
							enableSemanticCache: input.enableSemanticCache,
							semanticThreshold: input.semanticThreshold,
							enablePromptCache: input.enablePromptCache,
							promptCacheTTL: input.promptCacheTTL,
						},
					});

					// Create cluster models
					await tx.clusterModel.createMany({
						data: input.models.map((model) => ({
							clusterId: newCluster.id,
							provider: model.provider,
							modelName: model.modelName,
							priority: model.priority,
						})),
					});

					// Return cluster with models
					return await tx.lLMCluster.findUnique({
						where: { id: newCluster.id },
						include: {
							models: {
								orderBy: { priority: "asc" },
							},
						},
					});
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, input.projectId);
				await invalidateProjectCache(cacheKey, input.projectId);

				return cluster;
			} catch (error) {
				console.error("Error creating LLM cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to create LLM cluster",
				});
			}
		}),

	// Update an LLM cluster
	update: publicProcedure
		.input(updateClusterSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find cluster first
				const cluster = await ctx.db.lLMCluster.findFirst({
					where: { id: input.id },
					include: { project: true },
				});

				if (!cluster) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Cluster not found",
					});
				}

				// Check project access
				const auth = await authenticateAndGetProject(ctx, {
					projectId: cluster.projectId,
					apiKey: input.apiKey,
				});

				// Atomic update
				const updatedCluster = await ctx.db.$transaction(async (tx) => {
					// Update cluster
					return await tx.lLMCluster.update({
						where: { id: input.id },
						data: {
							...(input.name !== undefined && { name: input.name }),
							...(input.description !== undefined && {
								description: input.description,
							}),
							...(input.fallbackEnabled !== undefined && {
								fallbackEnabled: input.fallbackEnabled,
							}),
							...(input.fallbackMode !== undefined && { fallbackMode: input.fallbackMode }),
							...(input.enableCircuitBreaker !== undefined && {
								enableCircuitBreaker: input.enableCircuitBreaker,
							}),
							...(input.maxRetries !== undefined && { maxRetries: input.maxRetries }),
							...(input.timeoutMs !== undefined && { timeoutMs: input.timeoutMs }),
							...(input.costBias !== undefined && { costBias: input.costBias }),
							...(input.complexityThreshold !== undefined && {
								complexityThreshold: input.complexityThreshold,
							}),
							...(input.tokenThreshold !== undefined && {
								tokenThreshold: input.tokenThreshold,
							}),
							...(input.enableSemanticCache !== undefined && {
								enableSemanticCache: input.enableSemanticCache,
							}),
							...(input.semanticThreshold !== undefined && {
								semanticThreshold: input.semanticThreshold,
							}),
							...(input.enablePromptCache !== undefined && {
								enablePromptCache: input.enablePromptCache,
							}),
							...(input.promptCacheTTL !== undefined && {
								promptCacheTTL: input.promptCacheTTL,
							}),
							...(input.isActive !== undefined && { isActive: input.isActive }),
						},
						include: {
							models: {
								where: { isActive: true },
								orderBy: { priority: "asc" },
							},
						},
					});
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, cluster.projectId);
				await invalidateProjectCache(cacheKey, cluster.projectId);

				return updatedCluster;
			} catch (error) {
				console.error("Error updating LLM cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update LLM cluster",
				});
			}
		}),

	// Delete an LLM cluster
	delete: publicProcedure
		.input(
			z.object({
				id: z.string(),
				apiKey: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find cluster first
				const cluster = await ctx.db.lLMCluster.findFirst({
					where: { id: input.id },
					include: { project: true },
				});

				if (!cluster) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Cluster not found",
					});
				}

				// Check project access
				const auth = await authenticateAndGetProject(ctx, {
					projectId: cluster.projectId,
					apiKey: input.apiKey,
				});

				// Soft delete
				await ctx.db.lLMCluster.update({
					where: { id: input.id },
					data: { isActive: false },
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, cluster.projectId);
				await invalidateProjectCache(cacheKey, cluster.projectId);

				return { success: true };
			} catch (error) {
				console.error("Error deleting LLM cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to delete LLM cluster",
				});
			}
		}),

	// Add existing model to cluster
	addModel: publicProcedure
		.input(addModelSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find cluster first
				const cluster = await ctx.db.lLMCluster.findFirst({
					where: { id: input.clusterId, isActive: true },
					include: { project: true },
				});

				if (!cluster) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Cluster not found",
					});
				}

				// Check project access
				const auth = await authenticateAndGetProject(ctx, {
					projectId: cluster.projectId,
					apiKey: input.apiKey,
				});

				// Atomic model addition
				const model = await ctx.db.$transaction(async (tx) => {
					// Verify the model exists in our system
					const providerModel = await tx.providerModel.findFirst({
						where: {
							provider: { name: input.provider },
							name: input.modelName,
							isActive: true,
						},
					});

					if (!providerModel) {
						throw new TRPCError({
							code: "BAD_REQUEST",
							message: "Model not found or not available",
						});
					}

					// Check if model already exists in cluster
					const existingModel = await tx.clusterModel.findFirst({
						where: {
							clusterId: input.clusterId,
							provider: input.provider,
							modelName: input.modelName,
						},
					});

					if (existingModel) {
						throw new TRPCError({
							code: "CONFLICT",
							message: "Model already exists in this cluster",
						});
					}

					// Create cluster model
					return await tx.clusterModel.create({
						data: {
							clusterId: input.clusterId,
							provider: input.provider,
							modelName: input.modelName,
							priority: input.priority,
						},
					});
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, cluster.projectId);
				await invalidateProjectCache(cacheKey, cluster.projectId);

				return model;
			} catch (error) {
				console.error("Error adding model to cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to add model to cluster",
				});
			}
		}),

	// Remove model from cluster
	removeModel: publicProcedure
		.input(
			z.object({
				modelId: z.string(),
				apiKey: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find model first
				const model = await ctx.db.clusterModel.findFirst({
					where: { id: input.modelId },
					include: {
						cluster: { include: { project: true } },
					},
				});

				if (!model) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Model not found in cluster",
					});
				}

				// Check project access
				const auth = await authenticateAndGetProject(ctx, {
					projectId: model.cluster.projectId,
					apiKey: input.apiKey,
				});

				// Check business rule: don't allow removing last model
				const modelCount = await ctx.db.clusterModel.count({
					where: {
						clusterId: model.clusterId,
						isActive: true,
					},
				});

				if (modelCount <= 1) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot remove the last model from a cluster",
					});
				}

				// Soft delete
				await ctx.db.clusterModel.update({
					where: { id: input.modelId },
					data: { isActive: false },
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, model.cluster.projectId);
				await invalidateProjectCache(cacheKey, model.cluster.projectId);

				return { success: true };
			} catch (error) {
				console.error("Error removing model from cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to remove model from cluster",
				});
			}
		}),

	// Get available models
	getAvailableModels: publicProcedure
		.input(
			z.object({
				apiKey: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			try {
				// Basic auth check (no specific project required)
				if (input.apiKey) {
					const apiKeyRegex = /^sk-[A-Za-z0-9_-]+$/;
					if (!apiKeyRegex.test(input.apiKey)) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Invalid API key format",
						});
					}
					// Just verify key exists and is active
					const prefix = input.apiKey.slice(0, 11);
					const hash = crypto
						.createHash("sha256")
						.update(input.apiKey)
						.digest("hex");
					const record = await ctx.db.apiKey.findFirst({
						where: { keyPrefix: prefix, keyHash: hash, status: "active" },
					});
					if (!record || (record.expiresAt && record.expiresAt < new Date())) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Invalid or expired API key",
						});
					}
				} else {
					const clerkAuthResult = await getClerkAuth();
					if (!clerkAuthResult.userId) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Authentication required",
						});
					}
				}

				const models = await ctx.db.providerModel.findMany({
					where: {
						isActive: true,
						provider: { isActive: true },
					},
					include: {
						provider: true,
					},
					orderBy: [{ provider: { name: "asc" } }, { name: "asc" }],
				});

				return models;
			} catch (error) {
				console.error("Error fetching available models:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch available models",
				});
			}
		}),
});
