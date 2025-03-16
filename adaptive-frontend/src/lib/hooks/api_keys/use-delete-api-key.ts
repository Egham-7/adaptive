import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAPIKey } from "@/services/api_keys";

interface DeleteAPIKeyVariables {
  id: string;
  userId: string;
}

export const useDeleteAPIKey = () => {
  const queryClient = useQueryClient();

  return useMutation<void, Error, DeleteAPIKeyVariables>({
    mutationFn: ({ id }: DeleteAPIKeyVariables) => deleteAPIKey(id),
    onSuccess: (_, variables) => {
      // Invalidate and refetch the user's API keys list
      queryClient.invalidateQueries({
        queryKey: ["apiKeys", "user", variables.userId],
      });

      // Remove the deleted API key from the cache
      queryClient.removeQueries({ queryKey: ["apiKey", variables.id] });
    },
  });
};
