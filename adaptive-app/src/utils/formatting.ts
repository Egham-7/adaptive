// src/utils/formatting.ts
export function formatCurrencyWithDynamicPrecision(value: number): string {
	// Validate input
	if (!Number.isFinite(value)) {
		return "$0.00";
	}

	// Handle negative values
	const isNegative = value < 0;
	const absValue = Math.abs(value);

	const str = absValue.toString();
	const parts = str.split(".");
	const decimalPart = parts[1] || "";
	const significantDecimals = decimalPart.replace(/0+$/, "").length;
	const decimals = Math.min(Math.max(significantDecimals, 2), 8);
	const formatted = `$${absValue.toFixed(decimals)}`;
	return isNegative ? `-${formatted}` : formatted;
}
