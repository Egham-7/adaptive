import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createMessage } from "@/services/messages";
import { useAuth } from "@clerk/clerk-react";
import { BaseMessage } from "@/services/messages/types";

export const useCreateMessage = () => {
  const queryClient = useQueryClient();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  const createMessageMutation = useMutation({
    mutationFn: async ({
      convId,
      message,
    }: {
      convId: number;
      message: BaseMessage;
      provider?: string;
      model?: string;
    }) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      return createMessage(convId, message, token);
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["messages", variables.convId],
      });
    },
  });

  return createMessageMutation;
};
