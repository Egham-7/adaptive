import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createMessage } from "@/services/messages";
import { Message } from "@/services/llms/types";
import { useAuth } from "@clerk/clerk-react";

export const useCreateMessage = () => {
  const queryClient = useQueryClient();

  const { getToken, isLoaded, isSignedIn } = useAuth();

  const createMessageMutation = useMutation({
    mutationFn: async ({
      convId,
      message,
    }: {
      convId: number;
      message: Omit<Message, "id">;
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
