import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { z } from "zod";
import {
	authenticateAndGetProject,
	validateAndAuthenticateApiKey,
} from "@/lib/auth-utils";
import { upsertModelCapability } from "@/lib/model-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	addProviderModelSchema,
	createProviderSchema,
	getAllProvidersSchema,
	providerByIdSchema,
	providerByNameSchema,
	updateProviderModelSchema,
	updateProviderSchema,
} from "@/types/provider-schemas";

type ProviderWithModels = Prisma.ProviderGetPayload<{
	include: {
		models: {
			include: {
				capabilities: true;
			};
		};
	};
}>;

export const providersRouter = createTRPCRouter({
	// Get all providers
	getAll: publicProcedure
		.input(getAllProvidersSchema)
		.query(async ({ ctx, input }): Promise<ProviderWithModels[]> => {
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

				// Build visibility-based where clause
				const orConditions: Prisma.ProviderWhereInput[] = [
					// System providers (always visible)
					{ visibility: "system" },
					// Community providers (always visible)
					{ visibility: "community" },
				];

				const whereClause: Prisma.ProviderWhereInput = {
					OR: orConditions,
				};

				// Add project/organization scoped providers if projectId provided
				if (input.projectId) {
					try {
						// Verify user/API key has access to the specified project
						await authenticateAndGetProject(ctx, {
							projectId: input.projectId,
							apiKey: input.apiKey,
						});

						// Get project with organization projects for organization-scoped providers
						const project = await ctx.db.project.findFirst({
							where: { id: input.projectId },
							include: { organization: { include: { projects: true } } },
						});

						if (project) {
							const orgProjectIds = project.organization.projects.map(
								(p) => p.id,
							);

							orConditions.push(
								// Project-scoped providers
								{ visibility: "project", projectId: input.projectId },
								// Organization-scoped providers (any project in the org)
								{
									visibility: "organization",
									projectId: { in: orgProjectIds },
								},
							);
						}
					} catch (authError) {
						// If authorization fails, silently exclude project/org providers
						// Only system and community providers will be returned
						console.warn(
							`Project access denied for projectId ${input.projectId}:`,
							authError,
						);
					}
				}

				const providers = await ctx.db.provider.findMany({
					where: whereClause,
					include: {
						models: {
							include: {
								capabilities: true,
							},
							orderBy: { name: "asc" },
						},
					},
					orderBy: { displayName: "asc" },
				});

				return providers as ProviderWithModels[];
			} catch (error) {
				console.error("Error fetching providers:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch providers",
				});
			}
		}),

	// Get provider by ID
	getById: publicProcedure
		.input(providerByIdSchema)
		.query(async ({ ctx, input }) => {
			try {
				// Basic auth check
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

				// Build visibility-based where clause for security
				const orConditions: Prisma.ProviderWhereInput[] = [
					// System providers (always visible)
					{ visibility: "system" },
					// Community providers (always visible)
					{ visibility: "community" },
				];

				// Add project-scoped access if projectId provided
				if (input.projectId) {
					try {
						// Verify user/API key has access to the specified project
						await authenticateAndGetProject(ctx, {
							projectId: input.projectId,
							apiKey: input.apiKey,
						});

						// Get project with organization projects for organization-scoped providers
						const project = await ctx.db.project.findFirst({
							where: { id: input.projectId },
							include: { organization: { include: { projects: true } } },
						});

						if (project) {
							const orgProjectIds = project.organization.projects.map(
								(p) => p.id,
							);

							orConditions.push(
								// Project-scoped providers
								{ visibility: "project", projectId: input.projectId },
								// Organization-scoped providers (any project in the org)
								{
									visibility: "organization",
									projectId: { in: orgProjectIds },
								},
							);
						}
					} catch (authError) {
						// If authorization fails, only return public providers
						console.warn(
							`Project access denied for projectId ${input.projectId}:`,
							authError,
						);
					}
				}

				const provider = await ctx.db.provider.findFirst({
					where: {
						id: input.id,
						OR: orConditions,
					},
					include: {
						models: {
							include: {
								capabilities: true,
							},
							orderBy: { name: "asc" },
						},
					},
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found or access denied",
					});
				}

				return provider;
			} catch (error) {
				if (error instanceof TRPCError) throw error;
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider",
				});
			}
		}),

	// Get provider by name
	getByName: publicProcedure
		.input(providerByNameSchema)
		.query(async ({ ctx, input }) => {
			try {
				// Basic auth check
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

				// Build visibility-based where clause for security
				const orConditions: Prisma.ProviderWhereInput[] = [
					// System providers (always visible)
					{ visibility: "system" },
					// Community providers (always visible)
					{ visibility: "community" },
				];

				// Add project-scoped access if projectId provided
				if (input.projectId) {
					try {
						// Verify user/API key has access to the specified project
						await authenticateAndGetProject(ctx, {
							projectId: input.projectId,
							apiKey: input.apiKey,
						});

						// Get project with organization projects for organization-scoped providers
						const project = await ctx.db.project.findFirst({
							where: { id: input.projectId },
							include: { organization: { include: { projects: true } } },
						});

						if (project) {
							const orgProjectIds = project.organization.projects.map(
								(p) => p.id,
							);

							orConditions.push(
								// Project-scoped providers
								{ visibility: "project", projectId: input.projectId },
								// Organization-scoped providers (any project in the org)
								{
									visibility: "organization",
									projectId: { in: orgProjectIds },
								},
							);
						}
					} catch (authError) {
						// If authorization fails, only return public providers
						console.warn(
							`Project access denied for projectId ${input.projectId}:`,
							authError,
						);
					}
				}

				const provider = await ctx.db.provider.findFirst({
					where: {
						name: input.name,
						OR: orConditions,
					},
					include: {
						models: {
							include: {
								capabilities: true,
							},
							orderBy: { name: "asc" },
						},
					},
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				return provider;
			} catch (error) {
				if (error instanceof TRPCError) throw error;
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider",
				});
			}
		}),

	// Create a new provider (atomic transaction)
	create: publicProcedure
		.input(createProviderSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - only authenticated users can create providers
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
				}

				// For project/organization visibility, projectId is required
				if (
					(input.visibility === "project" ||
						input.visibility === "organization") &&
					!input.projectId
				) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message:
							"Project ID is required for project and organization visibility",
					});
				}

				// Verify user has access to the specified project
				if (input.projectId) {
					await authenticateAndGetProject(ctx, {
						projectId: input.projectId,
					});
				}

				// Atomic transaction for provider creation
				const provider = await ctx.db.$transaction(async (tx) => {
					// Check if provider name already exists in the same project/visibility scope
					const whereClause: Prisma.ProviderWhereInput = {
						name: input.name,
					};

					// For project visibility, check within project
					if (input.visibility === "project") {
						whereClause.projectId = input.projectId;
					}
					// For organization visibility, check within organization
					else if (input.visibility === "organization") {
						// Get all project IDs in this organization
						const project = await tx.project.findUnique({
							where: { id: input.projectId },
							include: { organization: { include: { projects: true } } },
						});

						if (project) {
							const orgProjectIds = project.organization.projects.map(
								(p) => p.id,
							);
							whereClause.projectId = { in: orgProjectIds };
							whereClause.visibility = "organization";
						}
					}
					// For community visibility, check globally
					else if (input.visibility === "community") {
						whereClause.visibility = "community";
					}

					const existingProvider = await tx.provider.findFirst({
						where: whereClause,
					});

					if (existingProvider) {
						const scope =
							input.visibility === "project"
								? "project"
								: input.visibility === "organization"
									? "organization"
									: "community";
						throw new TRPCError({
							code: "CONFLICT",
							message: `Provider name '${input.name}' already exists in this ${scope}`,
						});
					}

					// Create provider
					const newProvider = await tx.provider.create({
						data: {
							projectId: input.projectId,
							name: input.name,
							displayName: input.displayName,
							description: input.description,
							visibility: input.visibility,
							baseUrl: input.baseUrl,
							authType: input.authType,
							authHeaderName: input.authHeaderName,
							healthEndpoint: input.healthEndpoint,
							rateLimitRpm: input.rateLimitRpm,
							timeoutMs: input.timeoutMs,
							retryConfig: input.retryConfig,
							headers: input.headers,
						},
					});

					// Create provider models with capabilities
					for (const model of input.models) {
						const createdModel = await tx.providerModel.create({
							data: {
								providerId: newProvider.id,
								name: model.name,
								displayName: model.displayName,
								inputTokenCost: model.inputTokenCost,
								outputTokenCost: model.outputTokenCost,
							},
						});

						// Create capabilities if provided
						if (model.capabilities) {
							await upsertModelCapability(
								tx,
								createdModel.id,
								model.capabilities,
							);
						}
					}

					// Return provider with models and capabilities
					return await tx.provider.findUnique({
						where: { id: newProvider.id },
						include: {
							models: {
								include: {
									capabilities: true,
								},
								orderBy: { name: "asc" },
							},
						},
					});
				});

				return provider;
			} catch (error) {
				console.error("Error creating provider:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to create provider",
				});
			}
		}),

	// Update a provider
	update: publicProcedure
		.input(updateProviderSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - support both API key and Clerk authentication
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

				// Find provider first
				const provider = await ctx.db.provider.findFirst({
					where: { id: input.id },
					include: { project: true },
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				// Verify user has access to the provider's project
				if (provider.projectId) {
					await authenticateAndGetProject(ctx, {
						projectId: provider.projectId,
						apiKey: input.apiKey,
					});
				}

				// Update provider
				const updatedProvider = await ctx.db.provider.update({
					where: { id: input.id },
					data: {
						...(input.displayName && { displayName: input.displayName }),
						...(input.description !== undefined && {
							description: input.description,
						}),
						...(input.visibility !== undefined && {
							visibility: input.visibility,
						}),
						...(input.baseUrl !== undefined && { baseUrl: input.baseUrl }),
						...(input.authType !== undefined && { authType: input.authType }),
						...(input.authHeaderName !== undefined && {
							authHeaderName: input.authHeaderName,
						}),
						...(input.apiKey !== undefined && {
							apiKey: input.apiKey,
						}),
						...(input.healthEndpoint !== undefined && {
							healthEndpoint: input.healthEndpoint,
						}),
						...(input.rateLimitRpm !== undefined && {
							rateLimitRpm: input.rateLimitRpm,
						}),
						...(input.timeoutMs !== undefined && {
							timeoutMs: input.timeoutMs,
						}),
						...(input.retryConfig !== undefined && {
							retryConfig: input.retryConfig,
						}),
						...(input.headers !== undefined && { headers: input.headers }),
					},
					include: {
						models: {
							include: {
								capabilities: true,
							},
							orderBy: { name: "asc" },
						},
					},
				});

				return updatedProvider;
			} catch (error) {
				console.error("Error updating provider:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update provider",
				});
			}
		}),

	// Delete a provider
	delete: publicProcedure
		.input(providerByIdSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - support both API key and Clerk authentication
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

				// Find provider first
				const provider = await ctx.db.provider.findFirst({
					where: { id: input.id },
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				// Check if provider is being used in any clusters
				const clusterProviders = await ctx.db.clusterProvider.findFirst({
					where: {
						providerId: provider.id,
					},
				});

				if (clusterProviders) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot delete provider that is being used in clusters",
					});
				}

				// Hard delete since we removed isActive field
				await ctx.db.provider.delete({
					where: { id: input.id },
				});

				return { success: true };
			} catch (error) {
				console.error("Error deleting provider:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to delete provider",
				});
			}
		}),

	// Add model to existing provider
	addModel: publicProcedure
		.input(addProviderModelSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - support both API key and Clerk authentication
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

				// Find provider first
				const provider = await ctx.db.provider.findFirst({
					where: { id: input.providerId },
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				// Atomic model addition
				const model = await ctx.db.$transaction(async (tx) => {
					// Check if model already exists in provider
					const existingModel = await tx.providerModel.findFirst({
						where: {
							providerId: input.providerId,
							name: input.name,
						},
					});

					if (existingModel) {
						throw new TRPCError({
							code: "CONFLICT",
							message: "Model already exists in this provider",
						});
					}

					// Create provider model
					const createdModel = await tx.providerModel.create({
						data: {
							providerId: input.providerId,
							name: input.name,
							displayName: input.displayName,
							inputTokenCost: input.inputTokenCost,
							outputTokenCost: input.outputTokenCost,
						},
					});

					// Create capabilities if provided
					if (input.capabilities) {
						await upsertModelCapability(
							tx,
							createdModel.id,
							input.capabilities,
						);
					}

					// Return model with capabilities
					return await tx.providerModel.findUnique({
						where: { id: createdModel.id },
						include: {
							capabilities: true,
						},
					});
				});

				return model;
			} catch (error) {
				console.error("Error adding model to provider:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to add model to provider",
				});
			}
		}),

	// Update provider model
	updateModel: publicProcedure
		.input(updateProviderModelSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - support both API key and Clerk authentication
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

				// Find model first
				const model = await ctx.db.providerModel.findFirst({
					where: { id: input.id },
					include: { provider: true },
				});

				if (!model) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Model not found",
					});
				}

				// Update model and capabilities in transaction
				const updatedModel = await ctx.db.$transaction(async (tx) => {
					// Update model
					const _model = await tx.providerModel.update({
						where: { id: input.id },
						data: {
							...(input.displayName && { displayName: input.displayName }),
							...(input.inputTokenCost !== undefined && {
								inputTokenCost: input.inputTokenCost,
							}),
							...(input.outputTokenCost !== undefined && {
								outputTokenCost: input.outputTokenCost,
							}),
						},
					});

					// Update or create capabilities if provided
					if (input.capabilities) {
						await upsertModelCapability(tx, input.id, input.capabilities);
					}

					// Return updated model with capabilities
					return await tx.providerModel.findUnique({
						where: { id: input.id },
						include: {
							capabilities: true,
						},
					});
				});

				return updatedModel;
			} catch (error) {
				console.error("Error updating provider model:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update provider model",
				});
			}
		}),

	// Remove model from provider
	removeModel: publicProcedure
		.input(
			z.object({
				modelId: z.string(),
				apiKey: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check - support both API key and Clerk authentication
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

				// Find model first
				const model = await ctx.db.providerModel.findFirst({
					where: { id: input.modelId },
					include: { provider: true },
				});

				if (!model) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Model not found",
					});
				}

				// Atomic transaction to prevent race conditions
				await ctx.db.$transaction(async (tx) => {
					// Check business rule: don't allow removing last model
					const modelCount = await tx.providerModel.count({
						where: {
							providerId: model.providerId,
						},
					});

					if (modelCount <= 1) {
						throw new TRPCError({
							code: "BAD_REQUEST",
							message: "Cannot remove the last model from a provider",
						});
					}

					// Check if model is being used in any clusters
					// With the new schema, we need to check:
					// 1. ClusterProviders using this provider with no config (uses all models)
					// 2. ProviderConfigs with no selectedModels (uses all models)
					// 3. ProviderConfigModels that specifically select this model

					// Check cluster providers with no config
					const clusterProvidersUsingAll = await tx.clusterProvider.findFirst({
						where: {
							providerId: model.providerId,
							configId: null, // No config means use all models
						},
					});

					// Check if any configs exist with no model selections (atomic check)
					const configsUsingAll = await tx.providerConfig.findFirst({
						where: {
							providerId: model.providerId,
							selectedModels: {
								none: {
									// No active selections means using all models
								},
							},
						},
					});

					// Check if this specific model is selected in any config
					const configsUsingSpecific = await tx.providerConfigModel.findFirst({
						where: {
							providerModelId: model.id,
						},
					});

					if (
						clusterProvidersUsingAll ||
						configsUsingAll ||
						configsUsingSpecific
					) {
						throw new TRPCError({
							code: "BAD_REQUEST",
							message: "Cannot remove model that is being used in clusters",
						});
					}

					// Hard delete since we removed isActive field
					await tx.providerModel.delete({
						where: { id: input.modelId },
					});
				});

				return { success: true };
			} catch (error) {
				console.error("Error removing model from provider:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to remove model from provider",
				});
			}
		}),
});
