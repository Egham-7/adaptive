import type { z } from "zod";
import type { Context } from "@/server/api/trpc";
import type { modelCapabilitySchema } from "@/types/provider-schemas";

// Helper to create or update ModelCapability in a transaction
export const upsertModelCapability = async (
	tx: Parameters<Parameters<Context["db"]["$transaction"]>[0]>[0],
	providerModelId: string,
	capabilities: z.infer<typeof modelCapabilitySchema>,
) => {
	const existingCapabilities = await tx.modelCapability.findUnique({
		where: { providerModelId },
	});

	const capabilityData = {
		description: capabilities.description,
		maxContextTokens: capabilities.maxContextTokens,
		maxOutputTokens: capabilities.maxOutputTokens,
		supportsFunctionCalling: capabilities.supportsFunctionCalling ?? false,
		languagesSupported: capabilities.languagesSupported,
		modelSizeParams: capabilities.modelSizeParams,
		latencyTier: capabilities.latencyTier,
		taskType: capabilities.taskType,
		complexity: capabilities.complexity,
	};

	if (existingCapabilities) {
		return await tx.modelCapability.update({
			where: { providerModelId },
			data: {
				...(capabilities.description !== undefined && {
					description: capabilities.description,
				}),
				...(capabilities.maxContextTokens !== undefined && {
					maxContextTokens: capabilities.maxContextTokens,
				}),
				...(capabilities.maxOutputTokens !== undefined && {
					maxOutputTokens: capabilities.maxOutputTokens,
				}),
				...(capabilities.supportsFunctionCalling !== undefined && {
					supportsFunctionCalling: capabilities.supportsFunctionCalling,
				}),
				...(capabilities.languagesSupported !== undefined && {
					languagesSupported: capabilities.languagesSupported,
				}),
				...(capabilities.modelSizeParams !== undefined && {
					modelSizeParams: capabilities.modelSizeParams,
				}),
				...(capabilities.latencyTier !== undefined && {
					latencyTier: capabilities.latencyTier,
				}),
				...(capabilities.taskType !== undefined && {
					taskType: capabilities.taskType,
				}),
				...(capabilities.complexity !== undefined && {
					complexity: capabilities.complexity,
				}),
			},
		});
	}

	return await tx.modelCapability.create({
		data: {
			providerModelId,
			...capabilityData,
		},
	});
};
