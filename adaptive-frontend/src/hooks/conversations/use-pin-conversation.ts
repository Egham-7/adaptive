import { useMutation, useQueryClient } from "@tanstack/react-query";
import { pinConversation } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";
import { toast } from "sonner";

export const usePinConversation = () => {
  const { getToken, isLoaded, isSignedIn } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      conversationId,
    }: {
      conversationId: number;
      isPinned: boolean;
    }) => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();
      if (!token) {
        throw new Error("User is not signed in");
      }

      return pinConversation(conversationId, token);
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      const message = variables.isPinned
        ? "Conversation unpinned successfully"
        : "Conversation pinned successfully";
      toast.success(message);
    },
    onError: () => {
      toast.error("Failed to update conversation pin status");
    },
  });
};
