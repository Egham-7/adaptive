// src/utils/formatting.ts
export function formatCurrencyWithDynamicPrecision(value: number): string {
	const str = value.toString();
	const parts = str.split(".");
	const decimalPart = parts[1] || "";
	const significantDecimals = decimalPart.replace(/0+$/, "").length;
	const decimals = Math.min(Math.max(significantDecimals, 2), 8);
	return `$${value.toFixed(decimals)}`;
}