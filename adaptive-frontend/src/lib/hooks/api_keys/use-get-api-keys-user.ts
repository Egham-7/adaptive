import { useQuery } from "@tanstack/react-query";
import { getAPIKeysByUserId } from "@/services/api_keys";
import { APIKey } from "@/services/api_keys/types";

export const useGetAPIKeysByUserId = (userId: string) => {
  return useQuery<APIKey[], Error>({
    queryKey: ["apiKeys", "user", userId],
    queryFn: () => getAPIKeysByUserId(userId),
    enabled: !!userId,
  });
};
