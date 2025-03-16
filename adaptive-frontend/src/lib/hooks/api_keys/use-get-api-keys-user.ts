import { useQuery } from "@tanstack/react-query";
import { getAPIKeysByUserId } from "@/services/api_keys";
import { APIKey } from "@/services/api_keys/types";
import { useAuth } from "@clerk/clerk-react";

export const useGetAPIKeysByUserId = (userId: string) => {
  const { getToken, isSignedIn, isLoaded } = useAuth();
  return useQuery<APIKey[], Error>({
    queryKey: ["apiKeys", "user", userId],
    queryFn: async () => {
      if (!isSignedIn || !isLoaded) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("Token is not available");
      }

      return getAPIKeysByUserId(userId, token);
    },
    enabled: !!userId,
  });
};
