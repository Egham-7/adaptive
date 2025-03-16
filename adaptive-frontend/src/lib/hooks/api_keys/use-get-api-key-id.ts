import { useQuery } from "@tanstack/react-query";
import { getAPIKeyById } from "@/services/api_keys";
import { APIKey } from "@/services/api_keys/types";

export const useGetAPIKeyById = (id: string) => {
  return useQuery<APIKey, Error>({
    queryKey: ["apiKey", id],
    queryFn: () => getAPIKeyById(id),
    enabled: !!id,
  });
};
