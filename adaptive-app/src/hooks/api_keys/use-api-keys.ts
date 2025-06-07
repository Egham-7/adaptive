import { api } from "@/trpc/react";

export const useApiKeys = () => {
  return api.api_keys.list.useQuery(undefined, {
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });
};
