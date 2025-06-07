import { api } from "@/trpc/react";

export const useVerifyApiKey = (apiKey?: string) => {
	if (!apiKey) {
		throw new Error("API key is required for verification");
	}
	return api.api_keys.verify.useQuery(
		{ apiKey: apiKey },
		{
			enabled: !!apiKey,
			staleTime: 0, // Always fresh for verification
			refetchOnWindowFocus: false,
			retry: false, // Don't retry failed verifications
		},
	);
};
