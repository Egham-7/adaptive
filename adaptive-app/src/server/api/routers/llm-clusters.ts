import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { z } from "zod";
import {
	authenticateAndGetProject,
	getCacheKey,
	validateAndAuthenticateApiKey,
} from "@/lib/auth";
import { invalidateProjectCache, withCache } from "@/lib/shared/cache";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	addProviderToClusterSchema,
	type ClusterWithProviders,
	clusterByNameParamsSchema,
	createClusterSchema,
	projectClusterParamsSchema,
	updateClusterSchema,
} from "@/types/clusters";

export const llmClustersRouter = createTRPCRouter({
	// Get all clusters for a project
	getByProject: publicProcedure
		.input(projectClusterParamsSchema)
		.query(async ({ ctx, input }): Promise<ClusterWithProviders[]> => {
			try {
				await authenticateAndGetProject(ctx, input);

				return await withCache(
					`clusters:project:${input.projectId}`,
					async () => {
						const clusters = await ctx.db.lLMCluster.findMany({
							where: {
								projectId: input.projectId,
							},
							include: {
								providers: {
									include: {
										provider: {
											include: {
												models: {
													include: {
														capabilities: true,
													},
													orderBy: { name: "asc" },
												},
											},
										},
										config: true,
									},
									orderBy: { createdAt: "asc" },
								},
							},
							orderBy: { createdAt: "desc" },
						});

						return clusters;
					},
					300, // 5 minutes cache
				);
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

	// Get cluster by name
	getByName: publicProcedure
		.input(clusterByNameParamsSchema)
		.query(async ({ ctx, input }): Promise<ClusterWithProviders | null> => {
			try {
				await authenticateAndGetProject(ctx, input);

				const cluster = await ctx.db.lLMCluster.findFirst({
					where: {
						name: input.name,
						projectId: input.projectId,
					},
					include: {
						providers: {
							include: {
								provider: {
									include: {
										models: {
											include: {
												capabilities: true,
											},
											orderBy: { name: "asc" },
										},
									},
								},
								config: true,
							},
							orderBy: { createdAt: "asc" },
						},
					},
				});

				return cluster;
			} catch (error) {
				console.error("Error fetching LLM cluster by name:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch LLM cluster",
				});
			}
		}),

	// Create new cluster
	create: publicProcedure
		.input(createClusterSchema)
		.mutation(async ({ ctx, input }): Promise<ClusterWithProviders> => {
			try {
				const auth = await authenticateAndGetProject(ctx, input);

				// Pre-validation outside transaction
				const existingCluster = await ctx.db.lLMCluster.findFirst({
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

				// Validate all providers exist and are accessible
				for (const provider of input.providers) {
					const providerExists = await ctx.db.provider.findFirst({
						where: {
							id: provider.providerId,
							OR: [
								{ visibility: "system" },
								{ visibility: "community" },
								{ projectId: input.projectId },
							],
						},
					});

					if (!providerExists) {
						throw new TRPCError({
							code: "NOT_FOUND",
							message: `Provider with ID ${provider.providerId} not found`,
						});
					}

					// Validate config if specified
					if (provider.configId) {
						const configExists = await ctx.db.providerConfig.findFirst({
							where: {
								id: provider.configId,
								projectId: input.projectId,
								providerId: provider.providerId,
							},
						});

						if (!configExists) {
							throw new TRPCError({
								code: "NOT_FOUND",
								message: `Provider config with ID ${provider.configId} not found`,
							});
						}
					}
				}

				// Atomic cluster creation
				const cluster = await ctx.db.$transaction(async (tx) => {
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

					// Create cluster providers
					await tx.clusterProvider.createMany({
						data: input.providers.map((provider) => ({
							clusterId: newCluster.id,
							providerId: provider.providerId,
							configId: provider.configId,
						})),
					});

					// Return cluster with providers
					return await tx.lLMCluster.findUnique({
						where: { id: newCluster.id },
						include: {
							providers: {
								include: {
									provider: {
										include: {
											models: {
												include: {
													capabilities: true,
												},
											},
										},
									},
									config: true,
								},
								orderBy: { createdAt: "asc" },
							},
						},
					});
				});

				if (!cluster) {
					throw new TRPCError({
						code: "INTERNAL_SERVER_ERROR",
						message: "Failed to create cluster",
					});
				}

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
		.mutation(async ({ ctx, input }): Promise<ClusterWithProviders> => {
			try {
				// Find cluster first to get project info
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

				// Check for name conflicts if name is being updated
				if (input.name && input.name !== cluster.name) {
					const existingCluster = await ctx.db.lLMCluster.findFirst({
						where: {
							projectId: cluster.projectId,
							name: input.name,
							id: { not: input.id },
						},
					});

					if (existingCluster) {
						throw new TRPCError({
							code: "CONFLICT",
							message: "Cluster name already exists in this project",
						});
					}
				}

				// Atomic update
				const updatedCluster = await ctx.db.$transaction(async (tx) => {
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
							...(input.fallbackMode !== undefined && {
								fallbackMode: input.fallbackMode,
							}),
							...(input.enableCircuitBreaker !== undefined && {
								enableCircuitBreaker: input.enableCircuitBreaker,
							}),
							...(input.maxRetries !== undefined && {
								maxRetries: input.maxRetries,
							}),
							...(input.timeoutMs !== undefined && {
								timeoutMs: input.timeoutMs,
							}),
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
							// Note: isActive field was removed from schema
						},
						include: {
							providers: {
								include: {
									provider: {
										include: {
											models: {
												include: {
													capabilities: true,
												},
											},
										},
									},
									config: true,
								},
								orderBy: { createdAt: "asc" },
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

				// Hard delete since isActive field was removed
				await ctx.db.lLMCluster.delete({
					where: { id: input.id },
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

	// Add provider to cluster
	addProvider: publicProcedure
		.input(addProviderToClusterSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find cluster first
				const cluster = await ctx.db.lLMCluster.findFirst({
					where: { id: input.clusterId },
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

				// Pre-validation outside transaction
				const provider = await ctx.db.provider.findFirst({
					where: {
						id: input.providerId,
						OR: [
							{ visibility: "system" },
							{ visibility: "community" },
							{ projectId: cluster.projectId },
						],
					},
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: `Provider with ID ${input.providerId} not found`,
					});
				}

				// Validate provider config if specified
				if (input.configId) {
					const providerConfig = await ctx.db.providerConfig.findFirst({
						where: {
							id: input.configId,
							projectId: cluster.projectId,
							providerId: input.providerId,
						},
					});

					if (!providerConfig) {
						throw new TRPCError({
							code: "NOT_FOUND",
							message: `Provider config with ID ${input.configId} not found`,
						});
					}
				}

				// Check if provider already exists in cluster
				const existingClusterProvider = await ctx.db.clusterProvider.findFirst({
					where: {
						clusterId: input.clusterId,
						providerId: input.providerId,
					},
				});

				if (existingClusterProvider) {
					throw new TRPCError({
						code: "CONFLICT",
						message: "Provider already exists in this cluster",
					});
				}

				// Atomic provider addition
				const clusterProvider = await ctx.db.$transaction(async (tx) => {
					return await tx.clusterProvider.create({
						data: {
							clusterId: input.clusterId,
							providerId: input.providerId,
							configId: input.configId,
						},
						include: {
							provider: {
								include: {
									models: {
										include: {
											capabilities: true,
										},
									},
								},
							},
							config: true,
						},
					});
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, cluster.projectId);
				await invalidateProjectCache(cacheKey, cluster.projectId);

				return clusterProvider;
			} catch (error) {
				console.error("Error adding provider to cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to add provider to cluster",
				});
			}
		}),

	// Remove provider from cluster
	removeProvider: publicProcedure
		.input(
			z.object({
				clusterProviderId: z.string(),
				apiKey: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Find cluster provider first
				const clusterProvider = await ctx.db.clusterProvider.findFirst({
					where: { id: input.clusterProviderId },
					include: {
						cluster: { include: { project: true } },
					},
				});

				if (!clusterProvider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found in cluster",
					});
				}

				// Check project access
				const auth = await authenticateAndGetProject(ctx, {
					projectId: clusterProvider.cluster.projectId,
					apiKey: input.apiKey,
				});

				// Check business rule: don't allow removing last provider
				const providerCount = await ctx.db.clusterProvider.count({
					where: {
						clusterId: clusterProvider.clusterId,
					},
				});

				if (providerCount <= 1) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot remove the last provider from a cluster",
					});
				}

				// Delete cluster provider
				await ctx.db.clusterProvider.delete({
					where: { id: input.clusterProviderId },
				});

				// Invalidate cache
				const cacheKey = getCacheKey(auth, clusterProvider.cluster.projectId);
				await invalidateProjectCache(
					cacheKey,
					clusterProvider.cluster.projectId,
				);

				return { success: true };
			} catch (error) {
				console.error("Error removing provider from cluster:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to remove provider from cluster",
				});
			}
		}),

	// Get available models
	getAvailableModels: publicProcedure
		.input(
			z.object({
				projectId: z.string().optional(),
				apiKey: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			try {
				// Basic auth check (no specific project required)
				if (input.apiKey) {
					await validateAndAuthenticateApiKey(input.apiKey, ctx.db);
				} else {
					const clerkAuthResult = await getClerkAuth();
					if (!clerkAuthResult.userId) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Authentication required",
						});
					}
				}

				// Build visibility-based provider filter
				const providerOrConditions: Prisma.ProviderWhereInput[] = [
					// System providers (always visible)
					{ visibility: "system" },
					// Community providers (always visible)
					{ visibility: "community" },
				];

				// Add project/organization scoped providers if projectId provided
				if (input.projectId) {
					// Get user's organization projects for organization-scoped providers
					const project = await ctx.db.project.findFirst({
						where: { id: input.projectId },
						include: { organization: { include: { projects: true } } },
					});

					if (project) {
						const orgProjectIds = project.organization.projects.map(
							(p) => p.id,
						);

						providerOrConditions.push(
							// Project-scoped providers
							{ visibility: "project", projectId: input.projectId },
							// Organization-scoped providers (any project in the org)
							{ visibility: "organization", projectId: { in: orgProjectIds } },
						);
					}
				}

				const providerWhereClause: Prisma.ProviderWhereInput = {
					OR: providerOrConditions,
				};

				const models = await ctx.db.providerModel.findMany({
					where: {
						provider: providerWhereClause,
					},
					include: {
						provider: true,
						capabilities: true,
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
