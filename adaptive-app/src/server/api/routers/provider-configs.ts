import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import {
	authenticateAndGetProject,
	decryptProviderApiKey,
	encryptProviderApiKey,
	validateAndAuthenticateApiKey,
} from "@/lib/auth-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
	createProviderConfigSchema,
	deleteProviderConfigSchema,
	getProviderConfigsSchema,
	providerConfigByIdSchema,
	updateProviderConfigSchema,
} from "@/types/providers";

type ProviderConfigWithProvider = Prisma.ProviderConfigGetPayload<{
	include: {
		provider: {
			include: {
				models: {
					include: {
						capabilities: true;
					};
				};
			};
		};
	};
}>;

export const providerConfigsRouter = createTRPCRouter({
	// Get all provider configs for a project
	getAll: publicProcedure
		.input(getProviderConfigsSchema)
		.query(async ({ ctx, input }): Promise<ProviderConfigWithProvider[]> => {
			try {
				// Auth check
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

				// Verify access to project
				await authenticateAndGetProject(ctx, {
					projectId: input.projectId,
					apiKey: input.apiKey,
				});

				// Build where clause
				const whereClause: Prisma.ProviderConfigWhereInput = {
					projectId: input.projectId,
				};

				// Filter by specific provider if requested
				if (input.providerId) {
					whereClause.providerId = input.providerId;
				}

				const configs = await ctx.db.providerConfig.findMany({
					where: whereClause,
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
					},
					orderBy: { createdAt: "desc" },
				});

				// Decrypt provider API keys for response
				const decryptedConfigs = configs.map((config) => ({
					...config,
					providerApiKey: decryptProviderApiKey(config.providerApiKey),
				}));

				return decryptedConfigs as ProviderConfigWithProvider[];
			} catch (error) {
				console.error("Error fetching provider configs:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider configs",
				});
			}
		}),

	// Get provider config by ID
	getById: publicProcedure
		.input(providerConfigByIdSchema)
		.query(async ({ ctx, input }) => {
			try {
				// Auth check
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

				const config = await ctx.db.providerConfig.findFirst({
					where: { id: input.id },
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
					},
				});

				if (!config) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider config not found",
					});
				}

				// Verify user has access to this project
				await authenticateAndGetProject(ctx, {
					projectId: config.projectId,
					apiKey: input.apiKey,
				});

				// Decrypt provider API key for response
				const decryptedConfig = {
					...config,
					providerApiKey: decryptProviderApiKey(config.providerApiKey),
				};

				return decryptedConfig;
			} catch (error) {
				if (error instanceof TRPCError) throw error;
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider config",
				});
			}
		}),

	// Create a new provider config
	create: publicProcedure
		.input(createProviderConfigSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check
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

				// Verify access to project
				await authenticateAndGetProject(ctx, {
					projectId: input.projectId,
					apiKey: input.apiKey,
				});

				// Verify provider exists and is accessible
				const provider = await ctx.db.provider.findFirst({
					where: { id: input.providerId },
				});

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				// Check if config already exists for this provider in this project
				const existingConfig = await ctx.db.providerConfig.findFirst({
					where: {
						projectId: input.projectId,
						providerId: input.providerId,
					},
				});

				if (existingConfig) {
					throw new TRPCError({
						code: "CONFLICT",
						message:
							"Configuration already exists for this provider in this project",
					});
				}

				// Encrypt the provider API key before storing
				const encryptedApiKey = encryptProviderApiKey(input.providerApiKey);

				// Create provider config
				const config = await ctx.db.providerConfig.create({
					data: {
						projectId: input.projectId,
						providerId: input.providerId,
						displayName: input.displayName,
						providerApiKey: encryptedApiKey,
						customHeaders: input.customHeaders,
						customSettings: input.customSettings,
					},
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
					},
				});

				// Decrypt provider API key for response
				const decryptedConfig = {
					...config,
					providerApiKey: decryptProviderApiKey(config.providerApiKey),
				};

				return decryptedConfig;
			} catch (error) {
				console.error("Error creating provider config:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to create provider config",
				});
			}
		}),

	// Update a provider config
	update: publicProcedure
		.input(updateProviderConfigSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check
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

				// Find config first
				const config = await ctx.db.providerConfig.findFirst({
					where: { id: input.id },
				});

				if (!config) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider config not found",
					});
				}

				// Verify user has access to this project
				await authenticateAndGetProject(ctx, {
					projectId: config.projectId,
					apiKey: input.apiKey,
				});

				// Update provider config
				const updatedConfig = await ctx.db.providerConfig.update({
					where: { id: input.id },
					data: {
						...(input.displayName !== undefined && {
							displayName: input.displayName,
						}),
						...(input.providerApiKey !== undefined && {
							providerApiKey: encryptProviderApiKey(input.providerApiKey),
						}),
						...(input.customHeaders !== undefined && {
							customHeaders: input.customHeaders,
						}),
						...(input.customSettings !== undefined && {
							customSettings: input.customSettings,
						}),
					},
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
					},
				});

				// Decrypt provider API key for response
				const decryptedConfig = {
					...updatedConfig,
					providerApiKey: decryptProviderApiKey(updatedConfig.providerApiKey),
				};

				return decryptedConfig;
			} catch (error) {
				console.error("Error updating provider config:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update provider config",
				});
			}
		}),

	// Delete a provider config
	delete: publicProcedure
		.input(deleteProviderConfigSchema)
		.mutation(async ({ ctx, input }) => {
			try {
				// Auth check
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

				// Find config first
				const config = await ctx.db.providerConfig.findFirst({
					where: { id: input.id },
				});

				if (!config) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider config not found",
					});
				}

				// Verify user has access to this project
				await authenticateAndGetProject(ctx, {
					projectId: config.projectId,
					apiKey: input.apiKey,
				});

				// Delete provider config
				await ctx.db.providerConfig.delete({
					where: { id: input.id },
				});

				return { success: true };
			} catch (error) {
				console.error("Error deleting provider config:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to delete provider config",
				});
			}
		}),
});
