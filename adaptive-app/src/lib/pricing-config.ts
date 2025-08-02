/**
 * Centralized token pricing configuration
 * All pricing constants should be defined here to ensure consistency
 * and ease of maintenance across the application
 */

export const TOKEN_PRICING = {
	// Base pricing per 1M tokens (in USD)
	INPUT_TOKEN_PRICE_PER_MILLION: 0.05,
	OUTPUT_TOKEN_PRICE_PER_MILLION: 0.15,

	// Helper functions for calculations
	calculateInputCost: (tokens: number): number => {
		return (tokens / 1_000_000) * TOKEN_PRICING.INPUT_TOKEN_PRICE_PER_MILLION;
	},

	calculateOutputCost: (tokens: number): number => {
		return (tokens / 1_000_000) * TOKEN_PRICING.OUTPUT_TOKEN_PRICE_PER_MILLION;
	},

	calculateTotalCost: (inputTokens: number, outputTokens: number): number => {
		return TOKEN_PRICING.calculateInputCost(inputTokens) + TOKEN_PRICING.calculateOutputCost(outputTokens);
	},

	// Format pricing for display
	formatPrice: (amount: number): string => {
		return `$${amount.toFixed(6)}`;
	},

	// Get pricing display strings
	getDisplayPricing: () => ({
		input: `$${TOKEN_PRICING.INPUT_TOKEN_PRICE_PER_MILLION}/1M tokens`,
		output: `$${TOKEN_PRICING.OUTPUT_TOKEN_PRICE_PER_MILLION}/1M tokens`,
	}),
} as const;

// Credit limits and validation
export const CREDIT_LIMITS = {
	MINIMUM_PURCHASE: 1.0, // $1.00 minimum
	MAXIMUM_PURCHASE: 10000.0, // $10,000 maximum
	MINIMUM_DEDUCTION: 0.000001, // Very small for micro-transactions
	MAXIMUM_DEDUCTION: 10000.0, // $10,000 maximum
} as const;