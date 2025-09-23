import { TRPCError } from "@trpc/server";
import type { Prisma } from "prisma/generated";
import { getOrganizationBalance, hasSufficientCredits } from "@/lib/credits";
import { findModelBySimilarity, hashApiKey } from "@/lib/server/usage-utils";
import type { CacheTier } from "@/types/cache";

/**
 * Type for API key with project and organization included
 */
export type ApiKeyWithProject = Prisma.ApiKeyGetPayload<{
	include: {
		project: {
			include: {
				organization: true;
			};
		};
	};
}>;

/**
 * Apply cache tier discount to base cost
 */
export function applyCacheTierDiscount(
	baseCost: number,
	cacheTier?: CacheTier,
): number {
	switch (cacheTier) {
		case "prompt_response":
			return 0; // Free for prompt_response cache hits
		case "semantic_exact":
		case "semantic_similar":
			return baseCost * 0.5; // 50% discount for semantic cache hits
		default:
			return baseCost;
	}
}

/**
 * Calculate provider cost based on token usage and pricing
 */
export function calculateProviderCost(
	providerModel: {
		inputTokenCost: { toNumber(): number };
		outputTokenCost: { toNumber(): number };
	} | null,
	promptTokens: number,
	completionTokens: number,
): number {
	if (!providerModel) return 0;

	return (
		(promptTokens * providerModel.inputTokenCost.toNumber() +
			completionTokens * providerModel.outputTokenCost.toNumber()) /
		1_000_000
	);
}

/**
 * Validate API key and return with project and organization data
 */
export async function validateApiKey(
	db: any,
	apiKey: string,
): Promise<ApiKeyWithProject> {
	const keyHash = hashApiKey(apiKey);

	const apiKeyRecord = await db.apiKey.findFirst({
		where: {
			keyHash,
			status: "active",
		},
		include: {
			project: {
				include: {
					organization: true,
				},
			},
		},
	});

	if (!apiKeyRecord) {
		throw new TRPCError({
			code: "FORBIDDEN",
			message: "Invalid API key",
		});
	}

	return apiKeyRecord;
}

/**
 * Get organization ID from API key with validation
 */
export function getOrganizationId(apiKey: ApiKeyWithProject): string {
	const organizationId = apiKey.project?.organizationId;

	if (!organizationId) {
		throw new TRPCError({
			code: "INTERNAL_SERVER_ERROR",
			message: "API key is not associated with an organization",
		});
	}

	return organizationId;
}

/**
 * Check if organization has sufficient credits
 */
export async function checkSufficientCredits(
	organizationId: string,
	creditCost: number,
): Promise<void> {
	if (creditCost <= 0) return; // Skip check for free operations

	const hasEnoughCredits = await hasSufficientCredits(
		organizationId,
		creditCost,
	);

	if (!hasEnoughCredits) {
		const currentBalance = await getOrganizationBalance(organizationId);
		throw new TRPCError({
			code: "PAYMENT_REQUIRED",
			message: `Insufficient credits. Required: $${creditCost.toFixed(4)}, Available: $${currentBalance.toFixed(4)}. Please purchase more credits.`,
		});
	}
}

/**
 * Get provider model with cost information
 */
export async function getProviderModel(
	db: any,
	provider: string | null,
	model: string | null,
): Promise<{
	inputTokenCost: { toNumber(): number };
	outputTokenCost: { toNumber(): number };
} | null> {
	if (!provider || !model) return null;

	try {
		const result = await findModelBySimilarity(db, model, provider);
		if (!result) return null;
		return {
			inputTokenCost: result.inputTokenCost,
			outputTokenCost: result.outputTokenCost,
		};
	} catch (error) {
		console.error("Error fetching provider model:", error);
		return null;
	}
}
