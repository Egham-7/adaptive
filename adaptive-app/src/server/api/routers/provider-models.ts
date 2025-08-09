import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { validateAndAuthenticateApiKey } from "@/lib/auth-utils";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import type { ProviderWithModels } from "@/types/prisma-types";

export const providerModelsRouter = createTRPCRouter({
	// Get models for a specific provider config
	getForConfig: publicProcedure
		.input(
			z.object({
				projectId: z.string(),
				providerId: z.string(),
				configId: z.string().optional(),
				apiKey: z.string(),
			}),
		)
		.query(async ({ ctx, input }) => {
			try {
				// Validate API key
				await validateAndAuthenticateApiKey(input.apiKey, ctx.db);

				// Get the provider config if specified
				if (input.configId) {
					const config = await ctx.db.providerConfig.findFirst({
						where: {
							id: input.configId,
							projectId: input.projectId,
							providerId: input.providerId,
						},
						include: {
							selectedModels: {
								include: {
									model: {
										include: {
											capabilities: true,
										},
									},
								},
								orderBy: { createdAt: "asc" },
							},
							provider: {
								include: {
									models: {
										include: {
											capabilities: true,
										},
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

					// If no specific models selected, return all provider models
					if (config.selectedModels.length === 0) {
						// Get all models for this provider
						const provider = await ctx.db.provider.findUnique({
							where: { id: config.providerId },
							include: {
								models: {
									include: {
										capabilities: true,
									},
								},
							},
						});
						return provider?.models ?? [];
					}

					// Otherwise, return only selected models
					return config.selectedModels.map((sm) => sm.model);
				}

				// If no config specified, return all models for the provider
				const provider = (await ctx.db.provider.findFirst({
					where: {
						id: input.providerId,
						OR: [
							{ visibility: "system" },
							{ visibility: "community" },
							{ projectId: input.projectId },
						],
					},
					include: {
						models: {
							include: {
								capabilities: true,
							},
						},
					},
				})) as ProviderWithModels;

				if (!provider) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message: "Provider not found",
					});
				}

				return provider.models;
			} catch (error) {
				console.error("Error fetching provider models:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider models",
				});
			}
		}),

	// Get all models for a provider
	getByProvider: publicProcedure
		.input(
			z.object({
				providerId: z.string(),
				projectId: z.string().optional(),
				apiKey: z.string().optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			try {
				// Validate API key if provided
				if (input.apiKey) {
					await validateAndAuthenticateApiKey(input.apiKey, ctx.db);
				}

				const models = await ctx.db.providerModel.findMany({
					where: {
						providerId: input.providerId,
					},
					include: {
						capabilities: true,
						provider: true,
					},
					orderBy: { name: "asc" },
				});

				return models;
			} catch (error) {
				console.error("Error fetching provider models:", error);
				if (error instanceof TRPCError) {
					throw error;
				}
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch provider models",
				});
			}
		}),
});
