import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateAPIKey } from "@/services/api_keys";
import { UpdateAPIKeyRequest, APIKey } from "@/services/api_keys/types";
import { useAuth } from "@clerk/clerk-react";
interface UpdateAPIKeyVariables {
  id: string;
  apiKeyData: UpdateAPIKeyRequest;
}

export const useUpdateAPIKey = () => {
  const queryClient = useQueryClient();
  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation<APIKey, Error, UpdateAPIKeyVariables>({
    mutationFn: async ({ id, apiKeyData }: UpdateAPIKeyVariables) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      return updateAPIKey(id, apiKeyData, token);
    },
    onSuccess: (data) => {
      // Invalidate the specific API key
      queryClient.invalidateQueries({ queryKey: ["apiKey", data.id] });

      // Also invalidate the user's API keys list
      queryClient.invalidateQueries({
        queryKey: ["apiKeys", "user", data.user_id],
      });
    },
  });
};
