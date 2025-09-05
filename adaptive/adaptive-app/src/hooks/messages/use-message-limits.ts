import { api } from "@/trpc/react";

// High limit for development to avoid message restrictions during testing
const DEV_MESSAGE_LIMIT = 999;

export function useMessageLimits() {
	const { data, isLoading, error } = api.messages.getRemainingDaily.useQuery();

	// Disable limits in development
	const isDevelopment = process.env.NODE_ENV === "development";

	return {
		isLoading,
		error,
		isUnlimited: isDevelopment || (data?.unlimited ?? false),
		remainingMessages: isDevelopment
			? DEV_MESSAGE_LIMIT
			: (data?.remaining ?? 0),
		hasReachedLimit: isDevelopment
			? false
			: data?.unlimited === false && (data?.remaining ?? 0) <= 0,
	};
}
