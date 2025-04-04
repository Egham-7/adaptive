import { useQuery } from "@tanstack/react-query";
import { getConversation } from "@/services/conversations";
import { useAuth } from "@clerk/clerk-react";

export const useConversation = (conversationId: number) => {
  const { getToken, isSignedIn, isLoaded } = useAuth();

  return useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: async () => {
      if (!isSignedIn || !isLoaded) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      return getConversation(conversationId, token);
    },
  });
};
