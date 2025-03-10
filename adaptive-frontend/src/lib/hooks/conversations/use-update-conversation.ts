import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateConversation } from "@/services/conversations";

export const useUpdateConversation = () => {
  const queryClient = useQueryClient();

  const updateConversationMutation = useMutation({
    mutationFn: ({ id, title }: { id: number; title: string }) =>
      updateConversation(id, title),
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
