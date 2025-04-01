import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateConversation } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";

export const useUpdateConversation = () => {
  const queryClient = useQueryClient();
  const { getToken, isSignedIn, isLoaded } = useAuth();

  const updateConversationMutation = useMutation({
    mutationFn: async ({ id, title }: { id: number; title: string }) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      return updateConversation(id, title, token);
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["conversation", variables.id],
      });
      queryClient.invalidateQueries({
        queryKey: ["conversations"],
      });
    },
  });

  return updateConversationMutation;
};
