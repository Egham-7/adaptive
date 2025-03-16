import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateAPIKey } from "@/services/api_keys";
import { UpdateAPIKeyRequest, APIKey } from "@/services/api_keys/types";

interface UpdateAPIKeyVariables {
  id: string;
  apiKeyData: UpdateAPIKeyRequest;
}

export const useUpdateAPIKey = () => {
  const queryClient = useQueryClient();

  return useMutation<APIKey, Error, UpdateAPIKeyVariables>({
    mutationFn: ({ id, apiKeyData }: UpdateAPIKeyVariables) =>
      updateAPIKey(id, apiKeyData),
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
