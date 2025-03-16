import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteAPIKey } from "@/services/api_keys";
import { useAuth } from "@clerk/clerk-react";

interface DeleteAPIKeyVariables {
  id: string;
  userId: string;
}

export const useDeleteAPIKey = () => {
  const queryClient = useQueryClient();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  return useMutation<void, Error, DeleteAPIKeyVariables>({
    mutationFn: async ({ id }: DeleteAPIKeyVariables) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      return deleteAPIKey(id, token);
    },
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
