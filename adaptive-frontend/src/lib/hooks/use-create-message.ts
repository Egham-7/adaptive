import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createMessage } from "@/services/messages";
import { Message } from "@/services/llms/types";

export const useCreateMessage = () => {
  const queryClient = useQueryClient();

  const createMessageMutation = useMutation({
    mutationFn: ({
      convId,
      message,
    }: {
      convId: number;
      message: Omit<Message, "id">;
    }) => createMessage(convId, message),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["messages", variables.convId],
      });
    },
  });

  return createMessageMutation;
};
