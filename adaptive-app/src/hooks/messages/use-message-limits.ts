import { api } from "@/trpc/react";

export function useMessageLimits() {
	const { data, isLoading, error } = api.messages.getRemainingDaily.useQuery();

	return {
		isLoading,
		error,
		isUnlimited: data?.unlimited ?? false,
		remainingMessages: data?.remaining ?? 0,
		hasReachedLimit: data?.unlimited === false && (data?.remaining ?? 0) <= 0,
	};
}
