import type { RouterInputs, RouterOutputs } from "./index";

// ---- Output Types ----

/**
 * The type for credit balance response.
 * Includes raw balance and formatted currency string.
 */
export type CreditBalance = RouterOutputs["credits"]["getBalance"];

/**
 * The type for comprehensive credit statistics.
 * Contains current balance, totals, and transaction summaries.
 */
export type CreditStats = RouterOutputs["credits"]["getStats"];

/**
 * The type for credit transaction history response.
 * Contains paginated transactions with formatting and metadata.
 */
export type CreditTransactionHistory =
	RouterOutputs["credits"]["getTransactionHistory"];

/**
 * The type for a single credit transaction item.
 * Includes amount, type, description, and related data.
 */
export type CreditTransactionItem =
	RouterOutputs["credits"]["getTransactionHistory"]["transactions"][number];

/**
 * The type for cost calculation response.
 * Contains cost breakdown for token usage.
 */
export type CreditCostCalculation = RouterOutputs["credits"]["calculateCost"];

/**
 * The type for credit sufficiency check response.
 * Indicates if organization has enough credits for a cost.
 */
export type CreditSufficiencyCheck =
	RouterOutputs["credits"]["checkSufficientCredits"];

/**
 * The type for promotional credit award response.
 * Contains success status and new balance.
 */
export type PromotionalCreditResponse =
	RouterOutputs["credits"]["awardPromotionalCredits"];

/**
 * The type for low balance status response.
 * Contains balance status and warning thresholds.
 */
export type LowBalanceStatus = RouterOutputs["credits"]["getLowBalanceStatus"];

/**
 * The type for Stripe checkout session response.
 * Contains checkout URL and session details.
 */
export type CheckoutSessionResponse =
	RouterOutputs["credits"]["createCheckoutSession"];

// ---- Input Types ----

/**
 * The type for the input when getting credit balance.
 */
export type CreditBalanceInput = RouterInputs["credits"]["getBalance"];

/**
 * The type for the input when getting credit stats.
 */
export type CreditStatsInput = RouterInputs["credits"]["getStats"];

/**
 * The type for the input when getting transaction history.
 */
export type CreditTransactionHistoryInput =
	RouterInputs["credits"]["getTransactionHistory"];

/**
 * The type for the input when calculating cost.
 */
export type CreditCostCalculationInput =
	RouterInputs["credits"]["calculateCost"];

/**
 * The type for the input when checking sufficient credits.
 */
export type CreditSufficiencyInput =
	RouterInputs["credits"]["checkSufficientCredits"];

/**
 * The type for the input when awarding promotional credits.
 */
export type PromotionalCreditInput =
	RouterInputs["credits"]["awardPromotionalCredits"];

/**
 * The type for the input when checking low balance status.
 */
export type LowBalanceStatusInput =
	RouterInputs["credits"]["getLowBalanceStatus"];

/**
 * The type for the input when creating a checkout session.
 */
export type CheckoutSessionInput =
	RouterInputs["credits"]["createCheckoutSession"];

// ---- Utility Types ----

/**
 * The transaction type values that a credit transaction can have.
 */
export type CreditTransactionType = CreditTransactionItem["type"];

/**
 * The balance status values for low balance warnings.
 */
export type BalanceStatus = LowBalanceStatus["status"];

/**
 * Core credit balance data (just the numeric value).
 */
export type CreditBalanceAmount = CreditBalance["balance"];

/**
 * Credit transaction metadata structure.
 */
export type CreditTransactionMetadata = CreditTransactionItem["metadata"];

/**
 * Credit cost breakdown for detailed pricing information.
 */
export type CreditCostBreakdown = CreditCostCalculation["breakdown"];
