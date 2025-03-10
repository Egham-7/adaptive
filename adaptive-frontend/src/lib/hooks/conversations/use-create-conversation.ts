import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createConversation } from "@/services/conversations";

export const useCreateConversation = () => {
  const queryClient = useQueryClient();

  const createConversationMutation = useMutation({
    mutationFn: createConversation,
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
