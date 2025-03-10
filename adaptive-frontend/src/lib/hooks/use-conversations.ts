import { useQuery } from "@tanstack/react-query";
import { getConversations } from "@/services/conversations";
import type { Conversation } from "@/services/conversations/types";

/**
 * Hook for fetching all conversations
 *
 * @param options - Optional query configuration options
 * @returns Query result with conversations data, loading state, and error
 */
export const useConversations = (options?: {
  enabled?: boolean;
  staleTime?: number;
  refetchInterval?: number | false;
}) => {
  return useQuery<Conversation[], Error>({
    queryKey: ["conversations"],
    queryFn: getConversations,
    staleTime: options?.staleTime ?? 5 * 60 * 1000, // Default 5 minutes
    enabled: options?.enabled ?? true,
    refetchInterval: options?.refetchInterval ?? false,
  });
};
