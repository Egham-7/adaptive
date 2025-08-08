import crypto from "node:crypto";
import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { z } from "zod";
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

				const providers = await ctx.db.provider.findMany({
					include: {
						models: {
							where: { isActive: true },
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
					const apiKeyRegex = /^sk-[A-Za-z0-9_-]+$/;
					if (!apiKeyRegex.test(input.apiKey)) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Invalid API key format",
						});
					}
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

				const provider = await ctx.db.provider.findFirst({
					where: { id: input.id, isActive: true },
					include: {
						models: {
							where: { isActive: true },
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

	// Get provider by name
	getByName: publicProcedure
		.input(providerByNameSchema)
		.query(async ({ ctx, input }) => {
			try {
				// Basic auth check
				if (input.apiKey) {
					const apiKeyRegex = /^sk-[A-Za-z0-9_-]+$/;
					if (!apiKeyRegex.test(input.apiKey)) {
						throw new TRPCError({
							code: "UNAUTHORIZED",
							message: "Invalid API key format",
						});
					}
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

				const provider = await ctx.db.provider.findFirst({
					where: { name: input.name, isActive: true },
					include: {
						models: {
							where: { isActive: true },
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

				// Atomic transaction for provider creation
				const provider = await ctx.db.$transaction(async (tx) => {
					// Check if provider name already exists
					const existingProvider = await tx.provider.findFirst({
						where: { name: input.name },
					});

					if (existingProvider) {
						throw new TRPCError({
							code: "CONFLICT",
							message: "Provider name already exists",
						});
					}

					// Create provider
					const newProvider = await tx.provider.create({
						data: {
							name: input.name,
							displayName: input.displayName,
							description: input.description,
							isActive: true,
						},
					});

					// Create provider models with capabilities
					for (const model of input.models) {
						const createdModel = await tx.providerModel.create({
							data: {
								providerId: newProvider.id,
								name: model.name,
								displayName: model.displayName,
								type: model.type,
								inputTokenCost: model.inputTokenCost,
								outputTokenCost: model.outputTokenCost,
								isActive: true,
							},
						});

						// Create capabilities if provided
						if (model.capabilities) {
							await tx.modelCapability.create({
								data: {
									providerModelId: createdModel.id,
									description: model.capabilities.description,
									maxContextTokens: model.capabilities.maxContextTokens,
									maxOutputTokens: model.capabilities.maxOutputTokens,
									supportsFunctionCalling:
										model.capabilities.supportsFunctionCalling ?? false,
									languagesSupported:
										model.capabilities.languagesSupported?.join(","),
									modelSizeParams: model.capabilities.modelSizeParams,
									latencyTier: model.capabilities.latencyTier,
									taskType: model.capabilities.taskType,
									complexity: model.capabilities.complexity,
								},
							});
						}
					}

					// Return provider with models and capabilities
					return await tx.provider.findUnique({
						where: { id: newProvider.id },
						include: {
							models: {
								where: { isActive: true },
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
				// Auth check
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
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

				// Update provider
				const updatedProvider = await ctx.db.provider.update({
					where: { id: input.id },
					data: {
						...(input.displayName && { displayName: input.displayName }),
						...(input.description !== undefined && {
							description: input.description,
						}),
						...(input.isActive !== undefined && { isActive: input.isActive }),
					},
					include: {
						models: {
							where: { isActive: true },
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
				// Auth check
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
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
				const clusterModels = await ctx.db.clusterModel.findFirst({
					where: {
						provider: provider.name,
						isActive: true,
					},
				});

				if (clusterModels) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot delete provider that is being used in clusters",
					});
				}

				// Soft delete
				await ctx.db.provider.update({
					where: { id: input.id },
					data: { isActive: false },
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
				// Auth check
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
				}

				// Find provider first
				const provider = await ctx.db.provider.findFirst({
					where: { id: input.providerId, isActive: true },
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
							type: input.type,
							inputTokenCost: input.inputTokenCost,
							outputTokenCost: input.outputTokenCost,
							isActive: true,
						},
					});

					// Create capabilities if provided
					if (input.capabilities) {
						await tx.modelCapability.create({
							data: {
								providerModelId: createdModel.id,
								description: input.capabilities.description,
								maxContextTokens: input.capabilities.maxContextTokens,
								maxOutputTokens: input.capabilities.maxOutputTokens,
								supportsFunctionCalling:
									input.capabilities.supportsFunctionCalling ?? false,
								languagesSupported:
									input.capabilities.languagesSupported?.join(","),
								modelSizeParams: input.capabilities.modelSizeParams,
								latencyTier: input.capabilities.latencyTier,
								taskType: input.capabilities.taskType,
								complexity: input.capabilities.complexity,
							},
						});
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
				// Auth check
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
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
							...(input.type && { type: input.type }),
							...(input.inputTokenCost !== undefined && {
								inputTokenCost: input.inputTokenCost,
							}),
							...(input.outputTokenCost !== undefined && {
								outputTokenCost: input.outputTokenCost,
							}),
							...(input.isActive !== undefined && { isActive: input.isActive }),
						},
					});

					// Update or create capabilities if provided
					if (input.capabilities) {
						const existingCapabilities = await tx.modelCapability.findUnique({
							where: { providerModelId: input.id },
						});

						if (existingCapabilities) {
							await tx.modelCapability.update({
								where: { providerModelId: input.id },
								data: {
									...(input.capabilities.description !== undefined && {
										description: input.capabilities.description,
									}),
									...(input.capabilities.maxContextTokens !== undefined && {
										maxContextTokens: input.capabilities.maxContextTokens,
									}),
									...(input.capabilities.maxOutputTokens !== undefined && {
										maxOutputTokens: input.capabilities.maxOutputTokens,
									}),
									...(input.capabilities.supportsFunctionCalling !==
										undefined && {
										supportsFunctionCalling:
											input.capabilities.supportsFunctionCalling,
									}),
									...(input.capabilities.languagesSupported !== undefined && {
										languagesSupported:
											input.capabilities.languagesSupported?.join(","),
									}),
									...(input.capabilities.modelSizeParams !== undefined && {
										modelSizeParams: input.capabilities.modelSizeParams,
									}),
									...(input.capabilities.latencyTier !== undefined && {
										latencyTier: input.capabilities.latencyTier,
									}),
									...(input.capabilities.taskType !== undefined && {
										taskType: input.capabilities.taskType,
									}),
									...(input.capabilities.complexity !== undefined && {
										complexity: input.capabilities.complexity,
									}),
								},
							});
						} else {
							await tx.modelCapability.create({
								data: {
									providerModelId: input.id,
									description: input.capabilities.description,
									maxContextTokens: input.capabilities.maxContextTokens,
									maxOutputTokens: input.capabilities.maxOutputTokens,
									supportsFunctionCalling:
										input.capabilities.supportsFunctionCalling ?? false,
									languagesSupported:
										input.capabilities.languagesSupported?.join(","),
									modelSizeParams: input.capabilities.modelSizeParams,
									latencyTier: input.capabilities.latencyTier,
									taskType: input.capabilities.taskType,
									complexity: input.capabilities.complexity,
								},
							});
						}
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
				// Auth check
				const clerkAuthResult = await getClerkAuth();
				if (!clerkAuthResult.userId) {
					throw new TRPCError({
						code: "UNAUTHORIZED",
						message: "Authentication required",
					});
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

				// Check business rule: don't allow removing last model
				const modelCount = await ctx.db.providerModel.count({
					where: {
						providerId: model.providerId,
						isActive: true,
					},
				});

				if (modelCount <= 1) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot remove the last model from a provider",
					});
				}

				// Check if model is being used in any clusters
				const clusterModels = await ctx.db.clusterModel.findFirst({
					where: {
						provider: model.provider.name,
						modelName: model.name,
						isActive: true,
					},
				});

				if (clusterModels) {
					throw new TRPCError({
						code: "BAD_REQUEST",
						message: "Cannot remove model that is being used in clusters",
					});
				}

				// Soft delete
				await ctx.db.providerModel.update({
					where: { id: input.modelId },
					data: { isActive: false },
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
