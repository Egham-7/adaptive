import { TRPCError } from "@trpc/server";
import type { CreditTransactionType } from "prisma/generated";
import { z } from "zod";
import { TOKEN_PRICING } from "@/lib/config/pricing";
import {
	awardPromotionalCredits,
	calculateCreditCost,
	getOrganizationBalance,
	getOrganizationCreditStats,
	getOrganizationTransactionHistory,
	hasSufficientCredits,
} from "@/lib/credits";
import { formatCurrency } from "@/lib/shared/currency";
import { stripe } from "@/lib/stripe/stripe";
import {
	createTRPCRouter,
	protectedProcedure,
	publicProcedure,
} from "@/server/api/trpc";

function hashApiKey(apiKey: string): string {
	// Simple hash function for API keys
	return Buffer.from(apiKey).toString("base64");
}

export const creditsRouter = createTRPCRouter({
	// Pre-flight credit check before API usage (used by backend services)
	checkCreditsBeforeUsage: publicProcedure
		.input(
			z.object({
				apiKey: z.string(),
				estimatedInputTokens: z
					.number()
					.min(0, "Input tokens must be non-negative"),
				estimatedOutputTokens: z
					.number()
					.min(0, "Output tokens must be non-negative"),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			// Hash the API key to find it in database
			const keyHash = hashApiKey(input.apiKey);

			// Verify API key and get organization
			const apiKey = await ctx.db.apiKey.findFirst({
				where: { keyHash },
				include: { project: { include: { organization: true } } },
			});

			if (!apiKey || !apiKey.project) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "Invalid API key",
				});
			}

			const organizationId = apiKey.project.organization.id;

			// Calculate estimated credit cost
			const estimatedCreditCost = calculateCreditCost(
				input.estimatedInputTokens,
				input.estimatedOutputTokens,
			);

			// Check if organization has sufficient credits
			const hasEnoughCredits = await hasSufficientCredits(
				organizationId,
				estimatedCreditCost,
			);

			if (!hasEnoughCredits) {
				const currentBalance = await getOrganizationBalance(organizationId);
				throw new TRPCError({
					code: "PAYMENT_REQUIRED",
					message: `Insufficient credits. Estimated cost: $${estimatedCreditCost.toFixed(
						4,
					)}, Available: $${currentBalance.toFixed(
						4,
					)}. Please purchase more credits.`,
				});
			}

			return {
				hasEnoughCredits: true,
				currentBalance: await getOrganizationBalance(organizationId),
				estimatedCost: estimatedCreditCost,
			};
		}),

	// Get organization's current credit balance
	getBalance: protectedProcedure
		.input(z.object({ organizationId: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const balance = await getOrganizationBalance(input.organizationId);
				return {
					balance,
					formattedBalance: formatCurrency(balance),
				};
			} catch (error) {
				// Check if it's a specific error we can handle
				if (error instanceof Error && error.message.includes("not found")) {
					throw new TRPCError({
						code: "NOT_FOUND",
						message:
							"Organization not found. Please make sure you have access to this organization.",
					});
				}

				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message:
						"Failed to fetch credit balance. Please try again or contact support if the issue persists.",
				});
			}
		}),

	// Get comprehensive credit statistics
	getStats: protectedProcedure
		.input(z.object({ organizationId: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const stats = await getOrganizationCreditStats(input.organizationId);
				return {
					...stats,
					// Add formatted versions for UI display
					formatted: {
						currentBalance: formatCurrency(stats.currentBalance),
						totalPurchased: formatCurrency(stats.totalPurchased),
						totalUsed: formatCurrency(stats.totalUsed),
					},
				};
			} catch (_error) {
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch credit statistics",
				});
			}
		}),

	// Get transaction history with pagination and filtering
	getTransactionHistory: protectedProcedure
		.input(
			z.object({
				organizationId: z.string(),
				limit: z.number().min(1).max(100).default(20),
				offset: z.number().min(0).default(0),
				type: z.enum(["purchase", "usage", "refund", "promotional"]).optional(),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const transactions = await getOrganizationTransactionHistory(
					input.organizationId,
					{
						limit: input.limit,
						offset: input.offset,
						type: input.type,
					},
				);

				// Format transactions for UI display
				const formattedTransactions = transactions.map((transaction) => {
					const amount = transaction.amount;
					const balanceAfter = transaction.balanceAfter;
					return {
						...transaction,
						formattedAmount: amount.greaterThanOrEqualTo(0)
							? `+${formatCurrency(amount).substring(1)}` // Remove $ and add +
							: `-${formatCurrency(amount.abs()).substring(1)}`, // Remove $ and add -
						formattedBalance: formatCurrency(balanceAfter),
						// Add readable descriptions for different transaction types
						readableType: (() => {
							const typeMap: Record<CreditTransactionType, string> = {
								purchase: "Credit Purchase",
								usage: "API Usage",
								refund: "Refund",
								promotional: "Promotional Credit",
							};
							return typeMap[transaction.type];
						})(),
					};
				});

				return {
					transactions: formattedTransactions,
					hasMore: transactions.length === input.limit,
					nextOffset: input.offset + transactions.length,
				};
			} catch (_error) {
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to fetch transaction history",
				});
			}
		}),

	// Calculate cost for a hypothetical API request (for preview/estimation)
	calculateCost: protectedProcedure
		.input(
			z.object({
				inputTokens: z.number().min(0),
				outputTokens: z.number().min(0),
			}),
		)
		.query(async ({ input }) => {
			const cost = calculateCreditCost(input.inputTokens, input.outputTokens);

			return {
				cost,
				formattedCost: formatCurrency(cost),
				breakdown: {
					inputCost: TOKEN_PRICING.calculateInputCost(input.inputTokens),
					outputCost: TOKEN_PRICING.calculateOutputCost(input.outputTokens),
					inputTokens: input.inputTokens,
					outputTokens: input.outputTokens,
					totalTokens: input.inputTokens + input.outputTokens,
				},
			};
		}),

	// Check if user has sufficient credits for a cost
	checkSufficientCredits: protectedProcedure
		.input(
			z.object({
				organizationId: z.string(),
				requiredAmount: z.number().min(0),
			}),
		)
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const hasSufficient = await hasSufficientCredits(
					input.organizationId,
					input.requiredAmount,
				);
				const currentBalance = await getOrganizationBalance(
					input.organizationId,
				);

				return {
					hasSufficientCredits: hasSufficient,
					currentBalance,
					requiredAmount: input.requiredAmount,
					shortfall: hasSufficient ? 0 : input.requiredAmount - currentBalance,
				};
			} catch (_error) {
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to check credit sufficiency",
				});
			}
		}),

	// Award promotional credits (admin/system use)
	awardPromotionalCredits: protectedProcedure
		.input(
			z.object({
				organizationId: z.string(),
				amount: z.number().min(0.01),
				description: z.string().min(1),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const result = await awardPromotionalCredits(
					input.organizationId,
					userId,
					input.amount,
					input.description,
				);

				return {
					success: true,
					newBalance: result.newBalance,
					transaction: result.transaction,
					message: `Successfully awarded ${formatCurrency(
						input.amount,
					)} in promotional credits`,
				};
			} catch (error) {
				if (
					error instanceof Error &&
					error.message.includes("already received")
				) {
					throw new TRPCError({
						code: "CONFLICT",
						message: "User has already received promotional credits",
					});
				}

				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to award promotional credits",
				});
			}
		}),

	// Get low balance warning threshold
	getLowBalanceStatus: protectedProcedure
		.input(z.object({ organizationId: z.string() }))
		.query(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				const balance = await getOrganizationBalance(input.organizationId);
				const LOW_BALANCE_THRESHOLD = 1.0; // $1.00
				const VERY_LOW_BALANCE_THRESHOLD = 0.1; // $0.10

				let status: "good" | "low" | "very_low" | "empty" = "good";
				let message = "";

				if (balance <= 0) {
					status = "empty";
					message =
						"Your credit balance is empty. Please purchase credits to continue using the API.";
				} else if (balance <= VERY_LOW_BALANCE_THRESHOLD) {
					status = "very_low";
					message = `Your credit balance is very low (${formatCurrency(
						balance,
					)}). Please consider purchasing credits soon.`;
				} else if (balance <= LOW_BALANCE_THRESHOLD) {
					status = "low";
					message = `Your credit balance is low (${formatCurrency(
						balance,
					)}). Consider purchasing credits.`;
				}

				return {
					balance,
					status,
					message,
					thresholds: {
						low: LOW_BALANCE_THRESHOLD,
						veryLow: VERY_LOW_BALANCE_THRESHOLD,
					},
				};
			} catch (_error) {
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to check balance status",
				});
			}
		}),

	// Create Stripe checkout session for credit purchase
	createCheckoutSession: protectedProcedure
		.input(
			z.object({
				organizationId: z.string(),
				amount: z.number().min(1).max(10000), // $1 minimum, $10,000 maximum
				successUrl: z.string(),
				cancelUrl: z.string(),
			}),
		)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.userId;

			if (!userId) {
				throw new TRPCError({
					code: "UNAUTHORIZED",
					message: "User ID not found in context",
				});
			}

			try {
				console.log("🛒 Creating checkout session.");

				// Get or create Stripe customer
				let customerId: string;

				// Check if user already has a Stripe customer ID
				const existingSubscription = await ctx.db.subscription.findUnique({
					where: { userId },
					select: { stripeCustomerId: true },
				});

				if (existingSubscription?.stripeCustomerId) {
					customerId = existingSubscription.stripeCustomerId;
				} else {
					// Create new Stripe customer
					const customer = await stripe.customers.create({
						metadata: {
							userId,
							type: "api_credit_customer",
						},
					});
					customerId = customer.id;

					// Store customer ID for future use
					await ctx.db.subscription.upsert({
						where: { userId },
						create: {
							userId,
							stripeCustomerId: customerId,
							status: "incomplete",
						},
						update: {
							stripeCustomerId: customerId,
						},
					});
				}

				// Create Stripe checkout session for one-time payment
				const session = await stripe.checkout.sessions.create({
					customer: customerId,
					payment_method_types: ["card"],
					mode: "payment", // One-time payment, not subscription
					line_items: [
						{
							price_data: {
								currency: "usd",
								product_data: {
									name: `API Credits - $${input.amount}`,
									description: `$${input.amount} in API credits for your account`,
								},
								unit_amount: Math.round(input.amount * 100), // Convert to cents
							},
							quantity: 1,
						},
					],
					success_url: input.successUrl,
					cancel_url: input.cancelUrl,
					metadata: {
						userId,
						organizationId: input.organizationId,
						creditAmount: input.amount.toString(),
						type: "credit_purchase",
					},
				});

				console.log("✅ Checkout session created.");
				return {
					checkoutUrl: session.url,
					sessionId: session.id,
					amount: input.amount,
					formattedAmount: formatCurrency(input.amount),
				};
			} catch (_error) {
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to create checkout session",
				});
			}
		}),
});
