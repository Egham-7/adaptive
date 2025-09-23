import type { Decimal } from "@prisma/client/runtime/library";

/**
 * Helper function for dynamic precision formatting based on actual decimal places needed
 * Handles both regular numbers and Prisma Decimal types
 */
export const formatCurrency = (amount: number | Decimal): string => {
	// Convert to string to analyze decimal places - handles both number and Decimal types
	const numStr = amount.toString();

	// Handle scientific notation (e.g., 1.5e-7)
	if (numStr.includes("e")) {
		// For Decimal objects, use toString() with precision; for numbers use toFixed
		return typeof amount === "number" ? `$${amount.toFixed(8)}` : `$${numStr}`;
	}

	// Split by decimal point
	const parts = numStr.split(".");

	// If no decimal part, show 2 decimal places for standard currency display
	if (parts.length === 1) {
		return typeof amount === "number"
			? `$${amount.toFixed(2)}`
			: `$${numStr}.00`;
	}

	// Count significant decimal places (excluding trailing zeros)
	const decimalPart = parts[1] || "";
	const significantDecimals = decimalPart.replace(/0+$/, "").length;

	// Show at least 2 decimal places, up to 8 based on actual precision needed
	const decimalPlaces = Math.max(2, Math.min(8, significantDecimals));

	if (typeof amount === "number") {
		return `$${amount.toFixed(decimalPlaces)}`;
	}
	// For Decimal objects, format the string representation maintaining precision
	const formattedDecimal = Number.parseFloat(numStr).toFixed(decimalPlaces);
	return `$${formattedDecimal}`;
};
