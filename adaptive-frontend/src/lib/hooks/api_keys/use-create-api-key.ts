import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createAPIKey } from "@/services/api_keys";
import {
  CreateAPIKeyRequest,
  CreateAPIKeyResponse,
} from "@/services/api_keys/types";

export const useCreateAPIKey = () => {
  const queryClient = useQueryClient();

  return useMutation<CreateAPIKeyResponse, Error, CreateAPIKeyRequest>({
    mutationFn: (apiKeyData: CreateAPIKeyRequest) => createAPIKey(apiKeyData),
    onSuccess: (variables) => {
      // Invalidate and refetch the user's API keys list
      queryClient.invalidateQueries({
        queryKey: ["apiKeys", "user", variables.api_key.user_id],
      });
    },
  });
};
