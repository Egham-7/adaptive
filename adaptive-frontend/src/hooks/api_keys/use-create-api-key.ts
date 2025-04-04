import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createAPIKey } from "@/services/api_keys";
import {
  CreateAPIKeyRequest,
  CreateAPIKeyResponse,
} from "@/services/api_keys/types";
import { useAuth } from "@clerk/clerk-react";

export const useCreateAPIKey = () => {
  const queryClient = useQueryClient();
  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation<CreateAPIKeyResponse, Error, CreateAPIKeyRequest>({
    mutationFn: async (apiKeyData: CreateAPIKeyRequest) => {
      if (!isLoaded) {
        throw new Error("Authentication is still loading");
      }

      if (!isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("Failed to get authentication token");
      }

      return createAPIKey(apiKeyData, token);
    },
    onSuccess: (data) => {
      // Invalidate and refetch the user's API keys list
      queryClient.invalidateQueries({
        queryKey: ["apiKeys", "user", data.api_key.user_id],
      });
    },
  });
};
