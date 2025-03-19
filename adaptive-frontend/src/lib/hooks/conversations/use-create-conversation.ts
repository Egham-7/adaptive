import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createConversation } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";

export const useCreateConversation = () => {
  const queryClient = useQueryClient();

  const { getToken, isSignedIn, isLoaded } = useAuth();

  const createConversationMutation = useMutation({
    mutationFn: async (title: string) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      return createConversation(token, title);
    },
    onSuccess: (newConversation) => {
      queryClient.invalidateQueries({
        queryKey: ["conversation", newConversation.id],
      });
      queryClient.invalidateQueries({
        queryKey: ["conversations"],
      });
    },
  });

  return createConversationMutation;
};
