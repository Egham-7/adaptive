import { useQuery } from "@tanstack/react-query";
import { getConversations } from "@/services/conversations";
import type { Conversation } from "@/services/conversations/types";
import { useAuth } from "@clerk/clerk-react";

/**
 * Hook for fetching all conversations
 *
 * @param options - Optional query configuration options
 * @returns Query result with conversations data, loading state, and error
 */
export const useConversations = () => {
  const { getToken, isSignedIn, isLoaded } = useAuth();
  return useQuery<Conversation[], Error>({
    queryKey: ["conversations"],
    queryFn: async () => {
      if (!isSignedIn || !isLoaded) {
        throw new Error("User is not signed in");
      }

      const token = await getToken();

      if (!token) {
        throw new Error("User is not signed in");
      }
      return getConversations(token);
    },
  });
};
