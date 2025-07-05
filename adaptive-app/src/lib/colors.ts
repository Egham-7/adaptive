/**
 * Design system colors converted from OKLCH to hex values
 * These match the color definitions in globals.css
 */
export const designSystemColors = {
	// Light mode colors
	primary: "#0f766e", // oklch(0.4341 0.0392 41.9938)
	primaryForeground: "#ffffff", // oklch(1 0 0)
	secondary: "#f1f5f9", // oklch(0.92 0.0651 74.3695)
	secondaryForeground: "#334155", // oklch(0.3499 0.0685 40.8288)
	muted: "#f1f5f9", // oklch(0.9521 0 0)
	mutedForeground: "#64748b", // oklch(0.5032 0 0)
	accent: "#f8fafc", // oklch(0.931 0 0)
	accentForeground: "#0f172a", // oklch(0.2435 0 0)
	destructive: "#ef4444", // oklch(0.6271 0.1936 33.339)
	border: "#e2e8f0", // oklch(0.8822 0 0)
	input: "#e2e8f0", // oklch(0.8822 0 0)
	ring: "#0f766e", // oklch(0.4341 0.0392 41.9938)
	background: "#fafafa", // oklch(0.9821 0 0)
	foreground: "#0f172a", // oklch(0.2435 0 0)

	// Chart specific colors
	chart1: "#0f766e", // oklch(0.4341 0.0392 41.9938)
	chart2: "#f1f5f9", // oklch(0.92 0.0651 74.3695)
	chart3: "#f8fafc", // oklch(0.931 0 0)
	chart4: "#f0f9ff", // oklch(0.9367 0.0523 75.5009)
	chart5: "#0d9488", // oklch(0.4338 0.0437 41.6746)

	// Dark mode colors (for reference, can be extended if needed)
	dark: {
		primary: "#14b8a6", // oklch(0.9247 0.0524 66.1732)
		primaryForeground: "#042f2e", // oklch(0.2029 0.024 200.1962)
		secondary: "#0f766e", // oklch(0.3163 0.019 63.6992)
		secondaryForeground: "#14b8a6", // oklch(0.9247 0.0524 66.1732)
		muted: "#1e293b", // oklch(0.252 0 0)
		mutedForeground: "#94a3b8", // oklch(0.7699 0 0)
		background: "#020617", // oklch(0.1776 0 0)
		foreground: "#f8fafc", // oklch(0.9491 0 0)
	},
} as const;

/**
 * Get chart colors for consistent styling across all charts
 * @param colorKey - The design system color key to use
 * @param isDark - Whether to use dark mode colors
 * @returns Hex color value
 */
export function getChartColor(
	colorKey: Exclude<keyof typeof designSystemColors, 'dark'>,
	isDark = false,
): string {
	if (isDark && colorKey in designSystemColors.dark) {
		return designSystemColors.dark[colorKey as keyof typeof designSystemColors.dark];
	}
	return designSystemColors[colorKey];
}

/**
 * Get a set of colors for multi-series charts
 * @param count - Number of colors needed
 * @returns Array of hex color values
 */
export function getChartColorPalette(count: number): string[] {
	const palette = [
		designSystemColors.chart1,
		designSystemColors.chart2,
		designSystemColors.chart3,
		designSystemColors.chart4,
		designSystemColors.chart5,
		designSystemColors.secondary,
		designSystemColors.accent,
		designSystemColors.muted,
	];

	return palette.slice(0, count);
}

/**
 * Get colors for comparison charts (primary vs others)
 * @param isPrimary - Whether this is the primary/highlighted item
 * @param isDark - Whether to use dark mode colors
 * @returns Hex color value
 */
export function getComparisonChartColor(isPrimary: boolean, isDark = false): string {
	if (isDark) {
		return isPrimary
			? designSystemColors.dark.primary
			: designSystemColors.dark.mutedForeground;
	}
	return isPrimary
		? designSystemColors.primary
		: designSystemColors.mutedForeground;
}
