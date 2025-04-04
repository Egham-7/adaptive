import { useQuery } from "@tanstack/react-query";
import { getMessages } from "@/services/messages";
import { useAuth } from "@clerk/clerk-react";

export const useConversationMessages = (conversationId: number) => {
  const { getToken, isLoaded, isSignedIn } = useAuth();
  return useQuery({
    queryKey: ["messages", conversationId],
    queryFn: async () => {
      if (!isLoaded || !isSignedIn) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }

      const dbMessages = await getMessages(conversationId, token);

      return dbMessages;
    },
  });
};
