import { api } from "@/trpc/react";

export function useMessageLimits() {
	const { data, isLoading, error } = api.messages.getRemainingDaily.useQuery();

	// Disable limits in development
	const isDevelopment = process.env.NODE_ENV === "development";

	return {
		isLoading,
		error,
		isUnlimited: isDevelopment || (data?.unlimited ?? false),
		remainingMessages: isDevelopment ? 999 : (data?.remaining ?? 0),
		hasReachedLimit: isDevelopment
			? false
			: data?.unlimited === false && (data?.remaining ?? 0) <= 0,
	};
}
