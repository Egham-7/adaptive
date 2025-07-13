import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";

export const modelPricingRouter = createTRPCRouter({
	getProviders: publicProcedure.query(async ({ ctx }) => {
		return await ctx.db.provider.findMany({
			where: { isActive: true },
			select: {
				id: true,
				name: true,
				displayName: true,
			},
			orderBy: { displayName: "asc" },
		});
	}),

	getModelsByProvider: publicProcedure
		.input(z.object({ providerId: z.string() }))
		.query(async ({ ctx, input }) => {
			return await ctx.db.providerModel.findMany({
				where: {
					providerId: input.providerId,
					isActive: true,
					type: "chat", // Only chat models for cost comparison
				},
				select: {
					id: true,
					name: true,
					displayName: true,
					inputTokenCost: true,
					outputTokenCost: true,
				},
				orderBy: { displayName: "asc" },
			});
		}),

	calculateCostComparison: publicProcedure
		.input(
			z.object({
				currentModelName: z.string(),
				currentProviderName: z.string(),
				compareModelId: z.string(),
				inputTokens: z.number(),
				outputTokens: z.number(),
			}),
		)
		.query(async ({ ctx, input }) => {
			// Helper function to find best matching model by similarity
			const findModelBySimilarity = async (
				modelName: string,
				providerName: string,
			) => {
				// First try exact match
				const exactMatch = await ctx.db.providerModel.findFirst({
					where: {
						name: modelName,
						provider: { name: providerName },
					},
					select: {
						name: true,
						displayName: true,
						inputTokenCost: true,
						outputTokenCost: true,
						provider: { select: { name: true, displayName: true } },
					},
				});

				if (exactMatch) return exactMatch;

				// If no exact match, get all models for the provider and find best similarity
				const allModels = await ctx.db.providerModel.findMany({
					where: {
						provider: { name: providerName },
						isActive: true,
					},
					select: {
						name: true,
						displayName: true,
						inputTokenCost: true,
						outputTokenCost: true,
						provider: { select: { name: true, displayName: true } },
					},
				});

				if (allModels.length === 0) return null;

				// Simple similarity scoring: check if model name contains the base name
				const baseModelName = modelName.toLowerCase();
				let bestMatch = allModels[0];
				let bestScore = 0;

				for (const model of allModels) {
					const modelDbName = model.name.toLowerCase();

					// Score based on:
					// 1. If the db model name is contained in the input model name
					// 2. If they share common prefixes
					let score = 0;

					if (baseModelName.includes(modelDbName)) {
						score += 10;
					} else if (modelDbName.includes(baseModelName)) {
						score += 8;
					}

					// Check for common prefixes (e.g., "gpt-4o" matches "gpt-4o-mini")
					const commonPrefix = getCommonPrefix(baseModelName, modelDbName);
					if (commonPrefix.length > 3) {
						score += commonPrefix.length;
					}

					if (score > bestScore) {
						bestScore = score;
						bestMatch = model;
					}
				}

				return bestScore > 0 ? bestMatch : null;
			};

			// Helper function to get common prefix
			const getCommonPrefix = (str1: string, str2: string): string => {
				let i = 0;
				while (i < str1.length && i < str2.length && str1[i] === str2[i]) {
					i++;
				}
				return str1.substring(0, i);
			};

			const [currentModel, compareModel] = await Promise.all([
				findModelBySimilarity(
					input.currentModelName,
					input.currentProviderName,
				),
				ctx.db.providerModel.findUnique({
					where: { id: input.compareModelId },
					select: {
						name: true,
						displayName: true,
						inputTokenCost: true,
						outputTokenCost: true,
						provider: {
							select: { name: true, displayName: true },
						},
					},
				}),
			]);

			if (!currentModel || !compareModel) {
				throw new Error("Model not found");
			}

			// Calculate costs (costs are per 1M tokens)
			const currentCost =
				(Number(currentModel.inputTokenCost) * input.inputTokens) / 1000000 +
				(Number(currentModel.outputTokenCost) * input.outputTokens) / 1000000;

			const compareCost =
				(Number(compareModel.inputTokenCost) * input.inputTokens) / 1000000 +
				(Number(compareModel.outputTokenCost) * input.outputTokens) / 1000000;

			const savings = currentCost - compareCost;
			console.log("Savings: ", savings);
			const savingsPercentage =
				currentCost > 0 ? (Math.abs(savings) / currentCost) * 100 : 0;

			return {
				currentModel: {
					name: currentModel.displayName,
					provider: currentModel.provider.displayName,
					cost: currentCost,
				},
				compareModel: {
					name: compareModel.displayName,
					provider: compareModel.provider.displayName,
					cost: compareCost,
				},
				savings,
				savingsPercentage,
				isMoreExpensive: savings < 0,
			};
		}),
});
