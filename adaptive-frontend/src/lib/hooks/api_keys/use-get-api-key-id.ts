import { useQuery } from "@tanstack/react-query";
import { getAPIKeyById } from "@/services/api_keys";
import { APIKey } from "@/services/api_keys/types";
import { useAuth } from "@clerk/clerk-react";

export const useGetAPIKeyById = (id: string) => {
  const { getToken, isSignedIn, isLoaded } = useAuth();
  return useQuery<APIKey, Error>({
    queryKey: ["apiKey", id],
    queryFn: async () => {
      if (!isSignedIn || !isLoaded) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      return getAPIKeyById(id, token);
    },
    enabled: !!id,
  });
};
