import { api } from "@/trpc/react";

export const useApiKey = (id: string, enabled = true) => {
  return api.api_keys.getById.useQuery(
    { id },
    {
      enabled: enabled && !!id,
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  );
};
